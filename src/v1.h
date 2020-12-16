#include "v0.h"

#include <mpi.h>

//Computes distributed all-kNN of points in X
knnresult distrAllkNN(double * X, int n, int d, int k){

    //Store the world rank and size
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    //Declare the MPI status and request for the asynchronous communications
    MPI_Status idx_status, dist_status;
    MPI_Request idx_request, dist_request;

    //Define the number of queries in each process
    int m = n/world_size;
    for(int i=0; i<n%world_size; i++){              //if the number of processes is not dividable with the number of elements
        if(world_rank == i ) m = n/world_size + 1;  //the first will receive one extra 
    }

    //Initiallize the arrays that will store the indices and distances of the knn 
    int *nidx = malloc(k * sizeof(int));
    double *ndist = malloc(k * sizeof(double));        

    //Declaire that every process exept from 0 will receive the knn's indices and distances of the previous process
    //And will store the data on the buffers nidx and ndist instead of an internal buffer
    if(world_rank != 0){
        MPI_Irecv(nidx, k, MPI_INT, world_rank - 1, 0, MPI_COMM_WORLD, &idx_request);
        MPI_Irecv(ndist, k, MPI_DOUBLE, world_rank - 1, 1, MPI_COMM_WORLD, &dist_request);
    }
    
    knnresult knn;
    //Copy the points from X
    if(world_rank < n%world_size){
        knn = kNN(X, X + world_rank*m*d, n, m, d, k);
    }else{
        knn = kNN(X, X + (world_rank*m + n%world_size)*d, n, m, d, k);
    }

    //Sort the k smallest distances
    k_select(knn.nidx, knn.ndist, k , k * m);
    
    //Initiialize the return struct
    knnresult final;
    final.k = k;
    final.m = 1;
    final.nidx = malloc(k * sizeof(int));
    final.ndist = malloc(k * sizeof(double));     

    //The first process will copy the knn to the return struct 
    if(world_rank == 0){
        memcpy(final.nidx, knn.nidx, k * sizeof(int));
        memcpy(final.ndist, knn.ndist, k * sizeof(double));
    }

    //If there is only one process running return the knn
    if(world_size == 1) return final;
    
    //If there are more
    if (world_rank != 0) {
        
        //Wait to receive the k nearest from the previous process
        MPI_Wait(&idx_request, &idx_status);
        MPI_Wait(&dist_request, &dist_status);

        //Save the k nearest in the return struct
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
        
        //If this is the last process return 
        if(world_rank == world_size -1){
            return final;
        } 
    }

    //Send the updated nearest k to the next process
    MPI_Isend(final.nidx, k, MPI_INT, (world_rank + 1) % world_size, 0, MPI_COMM_WORLD, &idx_request);
    MPI_Isend(final.ndist, k, MPI_DOUBLE, (world_rank + 1) % world_size, 1, MPI_COMM_WORLD, &dist_request);

    //Finalize MPI and return an empty struct
    knnresult kn;
    return kn;
}
