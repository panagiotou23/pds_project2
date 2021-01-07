#include "vp_tree.h"
#include "v1.h"

//Copies vpt to prev_vpt
void copy_vp_tree(vp_tree *vpt, vp_tree *prev_vpt){

    if(prev_vpt->n == 0){

        prev_vpt->id = (int *)malloc(vpt->n * sizeof(int));
        prev_vpt->p = (double *)malloc(vpt->n * vpt->d * sizeof(double));
        prev_vpt->mu = (double *)malloc(vpt->n * sizeof(double));

        prev_vpt->left_cnt = (int *)malloc(vpt->n * sizeof(int));
        prev_vpt->right_cnt = (int *)malloc(vpt->n * sizeof(int));

    }else if(prev_vpt->n != vpt->n){

        prev_vpt->id = (int *)realloc(prev_vpt->id, vpt->n * sizeof(int));
        prev_vpt->p = (double *)realloc(prev_vpt->p, vpt->n * vpt->d * sizeof(double));
        prev_vpt->mu = (double *)realloc(prev_vpt->mu, vpt->n * sizeof(double));

        prev_vpt->left_cnt = (int *)realloc(prev_vpt->left_cnt, vpt->n * sizeof(int));
        prev_vpt->right_cnt = (int *)realloc(prev_vpt->right_cnt, vpt->n * sizeof(int));
    }

    prev_vpt->n = vpt->n;
    prev_vpt->d = vpt->d;

    prev_vpt->B = vpt->B;

    memcpy(prev_vpt->id, vpt->id, vpt->n * sizeof(int));
    memcpy(prev_vpt->p, vpt->p, vpt->n * vpt->d * sizeof(double));
    memcpy(prev_vpt->mu, vpt->mu, vpt->n * sizeof(double));

    memcpy(prev_vpt->left_cnt, vpt->left_cnt, vpt->n * sizeof(int));
    memcpy(prev_vpt->right_cnt, vpt->right_cnt, vpt->n * sizeof(int));

}

//Frees the allocated memory of vpt
void free_vp_tree(vp_tree vpt){

    free(vpt.id);
    free(vpt.p);
    free(vpt.mu);
    free(vpt.left_cnt);
    free(vpt.right_cnt);

}

//Calculates the kNN of its own points
knnresult get_my_kNN_2(vp_tree vpt, double *my_X, int m, int d, int k, int offset){

    knnresult my_knn;
    my_knn.k = k+1;
    my_knn.m = m;
    my_knn.nidx = (int *)malloc(m * (k+1) * sizeof(int));
    my_knn.ndist = (double *)malloc(m * (k+1) * sizeof(double));
    for(int i=0; i<m*(k+1); i++){
        my_knn.ndist[i] = INFINITY;
    }

    for(int i=0; i<m; i++)
        search(vpt, my_knn.nidx + i * (k+1), my_knn.ndist + i * (k+1), k+1, my_X + i*d, 0, 0, 1);
    

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

    return knn;
}

//Sends the vantage point tree to the next process & 
//receives the next one from the previous process
vp_tree get_other_vp_tree(vp_tree prev_vpt, int sender, int receiver){

    int flag = 0;
    MPI_Status status, statuses[10];
    MPI_Request requests[10];
    
    MPI_Isend(prev_vpt.id, prev_vpt.n, MPI_INT, receiver, 0, MPI_COMM_WORLD, &requests[0]);
    MPI_Isend(prev_vpt.p, prev_vpt.n * prev_vpt.d, MPI_DOUBLE, receiver, 1, MPI_COMM_WORLD, &requests[1]);
    MPI_Isend(prev_vpt.mu, prev_vpt.n, MPI_DOUBLE, receiver, 2, MPI_COMM_WORLD, &requests[2]);
    MPI_Isend(prev_vpt.left_cnt, prev_vpt.n, MPI_INT, receiver, 3, MPI_COMM_WORLD, &requests[3]);
    MPI_Isend(prev_vpt.right_cnt, prev_vpt.n, MPI_INT, receiver, 4, MPI_COMM_WORLD, &requests[4]);

    vp_tree other_vpt;
    other_vpt.d = prev_vpt.d;

    while(!flag) MPI_Iprobe(sender, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
    if(status.MPI_TAG == 1 || status.MPI_TAG == 2){
        MPI_Get_count( &status, MPI_DOUBLE, &other_vpt.n );        
        if(status.MPI_TAG == 1) other_vpt.n /= other_vpt.d;
    }else{
        MPI_Get_count( &status, MPI_INT, &other_vpt.n );  
    }
    
    other_vpt.id = (int *)malloc(other_vpt.n * sizeof(int));
    other_vpt.p = (double *)malloc(other_vpt.n * other_vpt.d * sizeof(double));
    other_vpt.mu = (double *)malloc(other_vpt.n * sizeof(double));

    other_vpt.left_cnt = (int *)malloc(other_vpt.n * sizeof(int));
    other_vpt.right_cnt = (int *)malloc(other_vpt.n * sizeof(int));

    other_vpt.B = prev_vpt.B;

    MPI_Irecv(other_vpt.id, other_vpt.n, MPI_INT, sender, 0, MPI_COMM_WORLD, &requests[5]);
    MPI_Irecv(other_vpt.p, other_vpt.n * other_vpt.d, MPI_DOUBLE, sender, 1, MPI_COMM_WORLD, &requests[6]);
    MPI_Irecv(other_vpt.mu, other_vpt.n, MPI_DOUBLE, sender, 2, MPI_COMM_WORLD, &requests[7]);
    MPI_Irecv(other_vpt.left_cnt, other_vpt.n, MPI_INT, sender, 3, MPI_COMM_WORLD, &requests[8]);
    MPI_Irecv(other_vpt.right_cnt, other_vpt.n, MPI_INT, sender, 4, MPI_COMM_WORLD, &requests[9]);

    MPI_Waitall(10, requests, statuses);

    return other_vpt;

}

//Calculates the distributed all-kNN of its points 
//by moving the vantage point trees in a ring
knnresult exchange_vp_trees(vp_tree vpt, double *my_X, knnresult knn){

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int receiver = world_rank + 1,
        sender = world_rank - 1;
                
    if(sender < 0) sender = world_size - 1;
    if(receiver == world_size) receiver = 0;

    vp_tree prev_vpt;
    prev_vpt.n = 0;
    copy_vp_tree(&vpt, &prev_vpt);
    free_vp_tree(vpt);

    for(int i=0; i<world_size-1; i++){

        vp_tree other_vpt = get_other_vp_tree(prev_vpt, sender, receiver);

        for(int i=0; i<knn.m; i++) 
            search(other_vpt, knn.nidx + i * knn.k, knn.ndist + i * knn.k, knn.k, my_X + i * other_vpt.d, 0, 0, 1);

        copy_vp_tree(&other_vpt, &prev_vpt);
        free_vp_tree(other_vpt);
    }

    free_vp_tree(prev_vpt);

    return knn;

}


//Computes distributed all-kNN of points in X
knnresult distrAllkNN_2(double * X, int n, int d, int k){

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int m = n/world_size;
    for(int i=0; i<n%world_size; i++){              
        if(world_rank == i ) m = n/world_size + 1;  
    }

    double *my_X = get_X(X, n, d, m);

    vp_tree vpt = make_vp_tree(my_X, m, d, k/10);
    
    int offset;
    if(world_rank < n%world_size){
        offset = world_rank * m;
    }else{
        offset = world_rank * m  + n%world_size;
    }

    for(int i=0; i<m; i++)
        vpt.id[i] += offset;

    knnresult knn = get_my_kNN_2(vpt, my_X, m, d, k, offset);

    if(world_size == 1){
        free(my_X);
        free_vp_tree(vpt);
        return knn;
    }

    knn = exchange_vp_trees(vpt, my_X, knn);    

    free(my_X);

    knnresult final = gather_final_kNN(n, m, k, knn);

    free(knn.nidx);
    free(knn.ndist);

    return final;
}
