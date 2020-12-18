#include "vp_tree.h"
#include "v1.h"
#include <mpi.h>

//Computes distributed all-kNN of points in X
knnresult distrAllkNN_2(double * X, int n, int d, int k){

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

    vp_tree vpt = make_vp_tree(my_X, m, d, k);
    
    knnresult knn;
    knn.k = k+1;
    knn.m = m;
    knn.nidx = malloc(m * (k+1) * sizeof(int));
    knn.ndist = malloc(m * (k+1) * sizeof(double));
    for(int i=0; i<m*(k+1); i++){
        knn.ndist[i] = INFINITY;
    }
    
    for(int i=0; i<m; i++) 
        search(vpt, knn.nidx + i * (k+1), knn.ndist + i * (k+1), k+1, my_X + i*d, 0, false, 1);

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
        MPI_Status status, statuses[10];
        MPI_Request requests[10];
        
        if(receiver == world_size) receiver = 0;

        MPI_Isend(vpt.id, m, MPI_INT, receiver, 0, MPI_COMM_WORLD, &requests[0]);
        MPI_Isend(vpt.p, m * d, MPI_DOUBLE, receiver, 1, MPI_COMM_WORLD, &requests[1]);
        MPI_Isend(vpt.mu, m, MPI_DOUBLE, receiver, 2, MPI_COMM_WORLD, &requests[2]);
        MPI_Isend(vpt.left_cnt, m, MPI_INT, receiver, 3, MPI_COMM_WORLD, &requests[3]);
        MPI_Isend(vpt.right_cnt, m, MPI_INT, receiver, 4, MPI_COMM_WORLD, &requests[4]);

        if(sender < 0) sender = world_size - 1;

        while(!flag) MPI_Iprobe(sender, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
        if(status.MPI_TAG == 1 || status.MPI_TAG == 2){
            MPI_Get_count( &status, MPI_DOUBLE, &other_m );        
            if(status.MPI_TAG == 1) other_m /= d;
        }else{
            MPI_Get_count( &status, MPI_INT, &other_m );  
        }

        vp_tree temp_vpt;
        temp_vpt.n = other_m;
        temp_vpt.d = d;
        
        temp_vpt.id = malloc(other_m * sizeof(int));
        temp_vpt.p = malloc(other_m * d * sizeof(double));
        temp_vpt.mu = malloc(other_m * sizeof(double));

        temp_vpt.left_cnt = malloc(other_m * sizeof(int));
        temp_vpt.right_cnt = malloc(other_m * sizeof(int));

        temp_vpt.B = k;

        MPI_Irecv(temp_vpt.id, other_m, MPI_INT, sender, 0, MPI_COMM_WORLD, &requests[5]);
        MPI_Irecv(temp_vpt.p, other_m * d, MPI_DOUBLE, sender, 1, MPI_COMM_WORLD, &requests[6]);
        MPI_Irecv(temp_vpt.mu, other_m, MPI_DOUBLE, sender, 2, MPI_COMM_WORLD, &requests[7]);
        MPI_Irecv(temp_vpt.left_cnt, other_m, MPI_INT, sender, 3, MPI_COMM_WORLD, &requests[8]);
        MPI_Irecv(temp_vpt.right_cnt, other_m, MPI_INT, sender, 4, MPI_COMM_WORLD, &requests[9]);

        MPI_Waitall(10, requests, statuses);
        
        knnresult temp_knn;
        temp_knn.k = k;
        temp_knn.m = m;
        temp_knn.nidx = malloc(m * k * sizeof(int));
        temp_knn.ndist = malloc(m * k * sizeof(double));
        for(int i=0; i<m*k; i++){
            temp_knn.ndist[i] = INFINITY;
        }
        
        for(int i=0; i<m; i++) 
            search(temp_vpt, temp_knn.nidx + i * k, temp_knn.ndist + i * k, k, my_X + i*d, 0, false, 1);

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

        sender--;
        receiver++;

    }

    return final;
}
