#include "v0.h"

#include <mpi.h>

//Computes distributed all-kNN of points in X
knnresult distrAllkNN_1(double * X, int n, int d, int k){

    //Store the world rank and size
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    //Define the number of queries in each process
    int m = n/world_size;
    for(int i=0; i<n%world_size; i++){              //if the number of processes is not dividable with the number of elements
        if(world_rank == i ) m = n/world_size + 1;  //the first will receive one extra 
    }

    int *sendcounts = (int *)malloc(world_size * sizeof(int));
    int *displs = (int *)malloc(world_size * sizeof(int));    

    for(int i=0; i<world_size; i++) sendcounts[i] = n/world_size * d;
    for(int i=0; i<n%world_size; i++) sendcounts[i] += d;                     
   
    int sum=0; 
    for(int i=0; i<world_size; i++){
        displs[i] = sum;
        sum += sendcounts[i];
    }

    double *my_X = (double *)malloc(m * d * sizeof(double));

    MPI_Scatterv(X, sendcounts, displs, 
                MPI_DOUBLE, my_X, m * d, 
                MPI_DOUBLE, 
                0, MPI_COMM_WORLD);

    free(sendcounts);
    free(displs);

    //Declare the variables used to time the function
    struct timespec ts_start;
    struct timespec ts_end;

    //Start the clock
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    knnresult my_knn = kNN(my_X, my_X, m, m, d, k + 1);

    //Stop the clock
    clock_gettime(CLOCK_MONOTONIC, &ts_end);

    //Calculate time 
    long v0_time = (ts_end.tv_sec - ts_start.tv_sec)* 1000000 + (ts_end.tv_nsec - ts_start.tv_nsec)/ 1000;

    if(world_rank == 0)
        printf("V0 time\n%ld us\n%f s\n\n", v0_time, v0_time*1e-6);

    int offset;
    if(world_rank < n%world_size){
        offset = world_rank * m;
    }else{
        offset = world_rank * m  + n%world_size;
    }
    for(int i=0; i<m; i++){
        for(int j=0; j<k+1; j++) {
            my_knn.nidx[j + i*(k+1)] += offset;
        }
    }
    
    for(int i=0; i<m; i++){
        for(int j=0; j<k+1; j++){
            if(my_knn.nidx[j + i * (k + 1)] == i + offset){
                memmove(my_knn.ndist + j + i * (k + 1), my_knn.ndist + j + 1 + i * (k + 1), (k - j) * sizeof(double));
                memmove(my_knn.nidx + j + i * (k + 1), my_knn.nidx + j + 1 + i * (k + 1), (k - j) * sizeof(int));
                break;
            }
        }
    }

    knnresult knn;
    knn.k = k;
    knn.m = m;
    knn.nidx = (int *)malloc(m * k * sizeof(int));
    knn.ndist = (double *)malloc(m * k * sizeof(double));
    for(int i=0; i<m; i++){
        memcpy(knn.nidx + i * k, my_knn.nidx + i * (k + 1), k * sizeof(int));
        memcpy(knn.ndist + i * k, my_knn.ndist + i * (k + 1) , k * sizeof(double));   
    }
    free(my_knn.nidx);
    free(my_knn.ndist);

    //If there is only one process running return the knn
    if(world_size == 1) return knn;
    
    int receiver = world_rank + 1,
        sender = world_rank - 1;
        
    if(sender < 0) sender = world_size - 1;
    if(receiver == world_size) receiver = 0;

    double *Z = (double *)malloc(m * d * sizeof(double));
    memcpy(Z, my_X, m * d * sizeof(double));
    int other_m = m;

    int points_owner = sender;

    for(int i=0; i<world_size-1; i++){
        
        int flag = 0;
        MPI_Status status;
        MPI_Request request;
        int prev_m = other_m;
        
        MPI_Isend(Z, other_m * d, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD, &request);

        while(!flag) MPI_Iprobe( sender, 0, MPI_COMM_WORLD, &flag, &status);
        MPI_Get_count( &status, MPI_DOUBLE, &other_m );        
        other_m /= d;

        double *other_X = (double *)malloc(other_m * d * sizeof(double));
        MPI_Recv(other_X , other_m * d, MPI_DOUBLE, sender, 0, MPI_COMM_WORLD, &status);
        
        //Start the clock
        clock_gettime(CLOCK_MONOTONIC, &ts_start);

        knnresult temp_knn = kNN(other_X, my_X, other_m, m, d, k);
        
        //Stop the clock
        clock_gettime(CLOCK_MONOTONIC, &ts_end);

        //Calculate time 
        v0_time = (ts_end.tv_sec - ts_start.tv_sec)* 1000000 + (ts_end.tv_nsec - ts_start.tv_nsec)/ 1000;

        if(world_rank == 0)
            printf("V0 time\n%ld us\n%f s\n\n", v0_time, v0_time*1e-6);

        int owner_offset;
        if(points_owner < n%world_size){
            owner_offset = points_owner * other_m;
        }else{
            owner_offset = points_owner * other_m + n%world_size;
        }

        for(int i=0; i<m; i++){
            for(int j=0; j<k; j++) {
                temp_knn.nidx[j + i*k] += owner_offset;
            }
        }

        points_owner--;
        if(points_owner < 0) points_owner = world_size - 1;

        for(int i=0; i<m; i++){

            int *nidx = (int *)malloc(2 * k * sizeof(int));
            double *ndist = (double *)malloc(2 * k * sizeof(double));

            memcpy(nidx, knn.nidx + i * k, k * sizeof(int));            
            memcpy(ndist, knn.ndist + i * k, k * sizeof(double));

            memcpy(nidx + k, temp_knn.nidx + i * k, k * sizeof(int));            
            memcpy(ndist + k, temp_knn.ndist + i * k, k * sizeof(double));
            
            quickselect(nidx, ndist, 0, (2 * k) - 1, k);

            memcpy(knn.nidx + i * k, nidx, k * sizeof(int));            
            memcpy(knn.ndist + i * k, ndist, k * sizeof(double));

            free(nidx);
            free(ndist);
        }

        MPI_Wait(&request, NULL);

        if(other_m != prev_m){
            Z = realloc(Z, other_m * d * sizeof(double));
        }
        memcpy(Z, other_X, other_m * d * sizeof(double));
        free(other_X);

    }
    
    free(my_X);
    free(Z);

    for(int i=0; i<m*k; i++) knn.ndist[i] = sqrt(knn.ndist[i]);

    int *recvcounts = (int *)malloc(world_size * sizeof(int));
    int *recvdispls = (int *)malloc(world_size * sizeof(int));    

    for(int i=0; i<world_size; i++) recvcounts[i] = n/world_size * k;
    for(int i=0; i<n%world_size; i++) recvcounts[i] += k;                     
   
    sum=0; 
    for(int i=0; i<world_size; i++){
        recvdispls[i] = sum;
        sum += recvcounts[i];
    }

    knnresult final;
    final.m = n;
    final.k = k;
    final.nidx = (int *)malloc(n * k * sizeof(int));
    final.ndist = (double *)malloc(n * k * sizeof(double));

    MPI_Gatherv(knn.nidx, m * k, MPI_INT, 
                final.nidx, recvcounts, recvdispls, 
                MPI_INT, 0, MPI_COMM_WORLD);
                
    MPI_Gatherv(knn.ndist, m * k, MPI_DOUBLE, 
                final.ndist, recvcounts, recvdispls, 
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    free(recvcounts);
    free(recvdispls);

    free(knn.nidx);
    free(knn.ndist);

    return final;
}
