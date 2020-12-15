#include "v0.h"

#include <mpi.h>


knnresult distrAllkNN(double * X, int n, int d, int k){

    MPI_Init(NULL, NULL);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    MPI_Status idx_status, dist_status;
    MPI_Request idx_request, dist_request;

    int m = n/world_size;
    for(int i=0; i<n%world_size; i++){
        if(world_rank == i ) m = n/world_size + 1;
    }


    int *nidx = (int *)malloc(k * sizeof(int));
    double *ndist = (double *)malloc(k * sizeof(double));        

    if(world_rank != 0){
        MPI_Irecv(nidx, k, MPI_INT, world_rank - 1, 0, MPI_COMM_WORLD, &idx_request);
        MPI_Irecv(ndist, k, MPI_DOUBLE, world_rank - 1, 1, MPI_COMM_WORLD, &dist_request);
    }


    double *Y = (double *)malloc(m * d * sizeof(double));
    // for(int i=0; i<m; i++){
    //     for(int j=0; j<d; j++){
    //         if(world_rank < n%world_size){
    //             Y[i][j] = X[i + world_rank*m][j];
    //         }else{
    //             Y[i][j] = X[i + world_rank*m + n%world_size][j];
    //         }            
    //     }
    // }

    // if(world_rank < n%world_size){
    //     for(int i=0; i<m; i++) memcpy(&Y[i], &X[i + world_rank*m], sizeof(X[0]));
    // }else{
    //     for(int i=0; i<m; i++) memcpy(&Y[i], &X[i + world_rank*m + n%world_size], sizeof(X[0]));
    // }
    
    if(world_rank < n%world_size){
        memcpy(Y, X + world_rank*m, m * d * sizeof(double));
    }else{
        memcpy(Y, X + world_rank*m + n%world_size, m * d * sizeof(double));
    }
    
    //The Variables used to time the function
    struct timespec ts_start;
    struct timespec ts_end;

    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    knnresult knn = kNN(X, Y, n, m, d, k);
    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    free(Y);

    long time = (ts_end.tv_sec - ts_start.tv_sec)* 1000000 + (ts_end.tv_nsec - ts_start.tv_nsec)/ 1000;
    printf("V0 Time \n%ld us\n%f s\n\n", time, (float)time*1e-6);

    k_select(knn.nidx, knn.ndist, k , k * m);
    knnresult final;

    final.k = k;
    final.m = 1;
    final.nidx = (int *)malloc(k * sizeof(int));
    final.ndist = (double *)malloc(k * sizeof(double));     

    if(world_rank == 0){
        memcpy(final.nidx, knn.nidx, k * sizeof(int));
        memcpy(final.ndist, knn.ndist, k * sizeof(double));
    }

    if(world_size == 1){
        MPI_Finalize();
        return final;
    } 
    
    if (world_rank != 0) {

        MPI_Wait(&idx_request, &idx_status);
        MPI_Wait(&dist_request, &dist_status);

        int i=0, 
            j=0;
        while(i + j < k){

            if(knn.ndist[i] < ndist[j]){
                final.ndist[i+j] = knn.ndist[i];
                final.nidx[i+j] = knn.nidx[i];
                i++;
            }else{
                final.ndist[i+j] = ndist[j];
                final.nidx[i+j] = nidx[j];
                j++;
            }
        }

        if(world_rank == world_size -1){
            MPI_Finalize();
            return final;
        } 
    }

    MPI_Isend(final.nidx, k, MPI_INT, (world_rank + 1) % world_size, 0, MPI_COMM_WORLD, &idx_request);
    MPI_Isend(final.ndist, k, MPI_DOUBLE, (world_rank + 1) % world_size, 1, MPI_COMM_WORLD, &dist_request);

    MPI_Finalize();
    knnresult kn;
    return kn;
}
