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

    double *my_X = malloc(m * d * sizeof(double));
    if(world_rank < n%world_size){
        memcpy(my_X, X + world_rank*m * d, m * d * sizeof(double));
    }else{
        memcpy(my_X, X + (world_rank*m + n%world_size) * d, m * d * sizeof(double));
    }

    knnresult knn = kNN(my_X, my_X, m, m, d, k + 1);
    
    for(int i=0; i<m; i++){
        for(int j=0; j<k+1; j++){
            if(knn.ndist[j + i * (k + 1)] == 0){
                SWAP(knn.ndist[(k + 1) * (i + 1) - 1], knn.ndist[j + i * (k + 1)], double);
                SWAP(knn.nidx[(k + 1) * (i + 1) - 1], knn.nidx[j + i * (k + 1)], int);
                break;
            }
        }
    }
    
    knnresult final;
    final.k = k;
    final.m = m;
    final.nidx = malloc(m * k * sizeof(int));
    final.ndist = malloc(m * k * sizeof(double));
    for(int i=0; i<m; i++){
        memcpy(final.nidx + i * k, knn.nidx + i * (k + 1), k * sizeof(int));
        memcpy(final.ndist + i * k, knn.ndist + i * (k + 1) , k * sizeof(double));   
    }


    if(world_rank < n%world_size){
        for(int i=0; i<m; i++){
            for(int j=0; j<k; j++) {
                final.nidx[j + i*k] += world_rank * m;
            }
        }
    }else{
        for(int i=0; i<m; i++){
            for(int j=0; j<k; j++) {
                final.nidx[j + i*k] += world_rank * m  + n%world_size;
            }
        }
    }

    //If there is only one process running return the knn
    if(world_size == 1) return final;
    
    int receiver = world_rank + 1,
        sender = world_rank - 1;
        
    for(int i=0; i<world_size-1; i++){

        int flag = 0;
        int other_m;
        MPI_Status status;
        MPI_Request request;
        
        if(receiver == world_size) receiver = 0;
        MPI_Isend(my_X, m * d, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD, &request);

        if(sender < 0) sender = world_size - 1;
                
        while(!flag) MPI_Iprobe( sender, 0, MPI_COMM_WORLD, &flag, &status);
        MPI_Get_count( &status, MPI_DOUBLE, &other_m );        
        other_m /= d;

        double *other_X = malloc(other_m * d * sizeof(double));
        MPI_Recv(other_X , other_m * d, MPI_DOUBLE, sender, 0, MPI_COMM_WORLD, &status);
        
        knnresult temp_knn = kNN(other_X, my_X, other_m, m, d, k);
        
        if(sender < n%world_size){
            for(int i=0; i<m; i++){
                for(int j=0; j<k; j++) {
                    temp_knn.nidx[j + i*k] += sender * other_m;
                }
            }
        }else{
            for(int i=0; i<m; i++){
                for(int j=0; j<k; j++) {
                    temp_knn.nidx[j + i*k] += sender * other_m + n%world_size;
                }
            }
        }

        for(int i=0; i<m; i++){

            int *nidx = malloc(2 * k * sizeof(int));
            double *ndist = malloc(2 * k * sizeof(double));

            memcpy(nidx, final.nidx + i * k, k * sizeof(int));            
            memcpy(ndist, final.ndist + i * k, k * sizeof(double));

            memcpy(nidx + k, temp_knn.nidx + i * k, k * sizeof(int));            
            memcpy(ndist + k, temp_knn.ndist + i * k, k * sizeof(double));
            
            quickselect(nidx, ndist, 0, (2 * k) - 1, k);

            memcpy(final.nidx + i * k, nidx, k * sizeof(int));            
            memcpy(final.ndist + i * k, ndist, k * sizeof(double));

            free(nidx);
            free(ndist);
        }

        free(other_X);

        sender--;
        receiver++;

    }

    return final;
}
