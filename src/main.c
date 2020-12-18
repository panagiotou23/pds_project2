#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include "v2.h"

#include <unistd.h>

/*****************************ADD READ MAT************************************/
/*************************CHANGE ALLKNN TO RANK 0*****************************/

int main(int argc, char *argv[]){

    //Initialiize MPI Commands
    MPI_Init(NULL, NULL);

    //Store the world rank and size
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    //Initialze rand()
    srand(time(0));

    //Set the number and dimensions of the data points
    int n = 20000;
    int d = 3;
    //Set the number of nearest neighbours
    int k = 10;

    int m = n/world_size;
    for(int i=0; i<n%world_size; i++){              //if the number of processes is not dividable with the number of elements
        if(world_rank == i ) m = n/world_size + 1;  //the first will receive one extra 
    }


    //Initialize the corpus data points
    double *X = malloc(n * d * sizeof(double));
    for(int i=0; i<n*d; i++) X[i] = (double)(rand()%100000)/1000;
    
    // if(world_rank == 0){
    //     printf("X\n");
    //     for(int i=0; i<n; i++){
    //         printf("%d\t", i);
    //         for(int j=0; j<d; j++) printf("%lf ", X[j + i*d]);
    //         printf("\n");
    //     }
    // }

    //Declare the variables used to time the function
    struct timespec ts_start;
    struct timespec ts_end;

    //Start the clock
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    // //Find the knn of the corpus data points
    knnresult knn1 = distrAllkNN_1(X, n, d, k);

    //Stop the clock
    clock_gettime(CLOCK_MONOTONIC, &ts_end);

    //Calculate time 
    long v1_time = (ts_end.tv_sec - ts_start.tv_sec)* 1000000 + (ts_end.tv_nsec - ts_start.tv_nsec)/ 1000;
    if(world_rank == 0) printf("V1 time\n%ld us\n%f s\n\n", v1_time, v1_time*1e-6);

    // sleep(world_rank);

    // for(int i=0; i<knn1.m; i++){
    //     // double *D;

    //     if(world_rank < n%world_size) {
    //         printf("For: %d\n", i + world_rank*m);
    //         // D = calc_Dcol(X, X, n, d, i + world_rank*m);
    //     }else{ 
    //         printf("For: %d\n", i + world_rank*m + n%world_size);
    //         // D = calc_Dcol(X, X, n, d, i + world_rank*m + n%world_size);
    //     }
    //     // printf("D\n");
    //     // for(int i=0; i<n; i++) printf("%d\t%.03f\n", i, D[i]);
    //     // printf("\n\n");
    //     for(int j=0; j<k; j++) printf("Dist: %.03f  \tIdx: %d\n", knn1.ndist[j + i * k], knn1.nidx[j + i * k]);
    //     printf("\n\n");
    // }
    // sleep(world_size);

    //Start the clock
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    
    knnresult knn2 = distrAllkNN_2(X, n, d, k);
    
    //Stop the clock
    clock_gettime(CLOCK_MONOTONIC, &ts_end);

    //Calculate time 
    long v2_time = (ts_end.tv_sec - ts_start.tv_sec)* 1000000 + (ts_end.tv_nsec - ts_start.tv_nsec)/ 1000;
    if(world_rank == 0) printf("V2 time\n%ld us\n%f s\n\n", v2_time, v2_time*1e-6);

    // sleep(world_rank);

    // for(int i=0; i<knn2.m; i++){
    //     double *D;

    //     if(world_rank < n%world_size) {
    //         printf("For: %d\n", i + world_rank*m);
    //         D = calc_Dcol(X, X, n, d, i + world_rank*m);
    //     }else{ 
    //         printf("For: %d\n", i + world_rank*m + n%world_size);
    //         D = calc_Dcol(X, X, n, d, i + world_rank*m + n%world_size);
    //     }
    //     // printf("D\n");
    //     // for(int i=0; i<n; i++) printf("%d\t%.03f\n", i, D[i]);
    //     // printf("\n\n");
    //     for(int j=0; j<k; j++) printf("Dist: %lf  \tIdx: %d\n", knn2.ndist[j + i * k], knn2.nidx[j + i * k]);
    //     printf("\n\n");
    // }

    free(X);
    MPI_Finalize();
    return 0;
}
