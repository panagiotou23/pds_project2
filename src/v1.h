#include <stdlib.h>

#include "v0.h"

#include <mpi.h>

//Distributes X to all processes
double *get_X(double *X, int n, int d, int m){

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

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

    return my_X;

}

//Calculates the kNN of its own points
knnresult get_my_kNN_1(double *X, int m, int d, int k, int offset){

    knnresult my_knn = kNN(X, X, m, m, d, k + 1);

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

    for(int i=0; i<m*k; i++) knn.ndist[i] = sqrt(knn.ndist[i]);

    free(my_knn.nidx);
    free(my_knn.ndist);


    return knn;
}

//Sends the previous set of points to the next process & 
//receives the next ones from the previous process
double *get_other_X(double *Z, int *other_m, int d, int sender, int receiver, MPI_Status *status, MPI_Request *request){
    int flag = 0;
    
    MPI_Isend(Z, *other_m * d, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD, request);

    while(!flag) MPI_Iprobe( sender, 0, MPI_COMM_WORLD, &flag, status);
    MPI_Get_count( status, MPI_DOUBLE, other_m );        
    *other_m /= d;

    double *other_X = (double *)malloc(*other_m * d * sizeof(double));
    MPI_Recv(other_X , *other_m * d, MPI_DOUBLE, sender, 0, MPI_COMM_WORLD, status);

    return other_X;
}

//Updates the kNN given the previous and new knnresult
knnresult update_KNN(knnresult knn, knnresult temp_knn, int m, int k){

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

    free(temp_knn.nidx);
    free(temp_knn.ndist);

    return knn;
}

//Calculates the distributed all-kNN of its points 
//by moving the sets of points in a ring
knnresult exchange_points(double *my_X, int n, int d, int m, int k, knnresult knn){

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int receiver = world_rank + 1,
        sender = world_rank - 1;
        
    if(sender < 0) sender = world_size - 1;
    if(receiver == world_size) receiver = 0;

    double *Z = (double *)malloc(m * d * sizeof(double));
    memcpy(Z, my_X, m * d * sizeof(double));
    int other_m = m;

    int points_owner = sender;

    for(int i=0; i<world_size-1; i++){
    
        int prev_m = other_m;
        MPI_Status status;
        MPI_Request request;
        
        double *other_X = get_other_X(Z, &other_m, d, sender, receiver, &status, &request);

        knnresult temp_knn = kNN(other_X, my_X, other_m, m, d, k);

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

        for(int i=0; i<m*k; i++) temp_knn.ndist[i] = sqrt(temp_knn.ndist[i]);

        points_owner--;
        if(points_owner < 0)
            points_owner = world_size - 1;

        knn = update_KNN(knn, temp_knn, m, k);

        MPI_Wait(&request, NULL);

        if(other_m != prev_m){
            Z = (double *)realloc(Z, other_m * d * sizeof(double));
        }
        memcpy(Z, other_X, other_m * d * sizeof(double));
        
        free(other_X);

    }
    
    free(Z);

    return knn;
}

//Gathers all the local kNN to process 0
knnresult gather_final_kNN(int n, int m, int k, knnresult knn){

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int *recvcounts = (int *)malloc(world_size * sizeof(int));
    int *recvdispls = (int *)malloc(world_size * sizeof(int));    

    for(int i=0; i<world_size; i++) recvcounts[i] = n/world_size * k;
    for(int i=0; i<n%world_size; i++) recvcounts[i] += k;                     
   
    int sum=0; 
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

    return final;

}


//Computes distributed all-kNN of points in X
knnresult distrAllkNN_1(double * X, int n, int d, int k){

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int m = n/world_size;
    for(int i=0; i<n%world_size; i++){              
        if(world_rank == i ) m = n/world_size + 1;  
    }

    double *my_X = get_X(X, n, d, m);

    int offset;
    if(world_rank < n%world_size){
        offset = world_rank * m;
    }else{
        offset = world_rank * m  + n%world_size;
    }

    knnresult knn = get_my_kNN_1(my_X, m, d, k, offset);

    if(world_size == 1){
        free(my_X);
        return knn;
    }

    knn = exchange_points(my_X, n, d, m, k, knn);
    free(my_X);

    knnresult final = gather_final_kNN(n, m, k, knn);

    free(knn.nidx);
    free(knn.ndist);

    return final;
}
