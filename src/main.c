#include <stdio.h>
#include <time.h>

#include "v2.h"

/*****************************ADD READ MAT************************************/

double *create_X(int n, int d){

    //Initialize the corpus data points
    double *X = malloc(n * d * sizeof(double));
    for(int i=0; i<n*d; i++) X[i] = (double)(rand()%100000)/1000;
    
    return X;
}

void print_X(double *X, int n, int d){
    printf("X\n");
    for(int i=0; i<n; i++){
        printf("%d\t", i);
        for(int j=0; j<d; j++) printf("%lf ", X[j + i*d]);
        printf("\n");
    }
}

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
    int n = 10e3;
    int d = 3;
    //Set the number of nearest neighbours
    int k = 100;

    double *X;

    if(!world_rank) {
        X = create_X(n, d);

        // print_X(X, n, d);
        // printf("\n\n");
    }

    //Declare the variables used to time the function
    struct timespec ts_start;
    struct timespec ts_end;

    //Start the clock
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    //Find the knn of the corpus data points
    knnresult knn1 = distrAllkNN_1(X, n, d, k);

    //Stop the clock
    clock_gettime(CLOCK_MONOTONIC, &ts_end);

    //Calculate time 
    long v1_time = (ts_end.tv_sec - ts_start.tv_sec)* 1000000 + (ts_end.tv_nsec - ts_start.tv_nsec)/ 1000;

    if(world_rank == 0){
        printf("V1 time\n%ld us\n%f s\n\n", v1_time, v1_time*1e-6);
        // for(int i=0; i<n; i++){
        //     double *D;
        //     printf("For: %d\n", i);
        //     D = calc_Dcol(X, X, n, d, i);
        //     printf("D\n");
        //     for(int i=0; i<n; i++) printf("%d\t%.03f\n", i, D[i]);
        //     printf("\n\n");
        //     for(int j=0; j<k; j++) printf("Dist: %.03f  \tIdx: %d\n", knn1.ndist[j + i * k], knn1.nidx[j + i * k]);
        //     printf("\n\n");
        // }
    }
    
    //Start the clock
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    
    knnresult knn2 = distrAllkNN_2(X, n, d, k);
    
    //Stop the clock
    clock_gettime(CLOCK_MONOTONIC, &ts_end);

    //Calculate time 
    long v2_time = (ts_end.tv_sec - ts_start.tv_sec)* 1000000 + (ts_end.tv_nsec - ts_start.tv_nsec)/ 1000;
    
    if(world_rank == 0){
        printf("V2 time\n%ld us\n%f s\n\n", v2_time, v2_time*1e-6);
        // for(int i=0; i<n; i++){
            // double *D;
            // printf("For: %d\n", i);
            // D = calc_Dcol(X, X, n, d, i);
            // printf("D\n");
            // for(int i=0; i<n; i++) printf("%d\t%.03f\n", i, D[i]);
            // printf("\n\n");
            // for(int j=0; j<k; j++) printf("Dist: %.03f  \tIdx: %d\n", knn2.ndist[j + i * k], knn2.nidx[j + i * k]);
            // printf("\n\n");
        // }

        int wrong_cnt = 0;
        for(int i=0; i<n; i++){

            k_select(knn1.nidx + i*k, knn1.ndist + i*k, k, k);
            k_select(knn2.nidx + i*k, knn2.ndist + i*k, k, k);
            
            for(int j=0; j<k; j++){
                if(knn1.nidx[j + i * k] != knn2.nidx[j + i * k]){
                    wrong_cnt++;
                    printf("%d\nDiff: %lf\n", i, knn1.ndist[j + i * k] - knn2.ndist[j + i * k]);
                    printf("Dist: %lf  \tIdx: %d\n", knn1.ndist[j + i * k], knn1.nidx[j + i * k]);
                    printf("Dist: %lf  \tIdx: %d\n", knn2.ndist[j + i * k], knn2.nidx[j + i * k]);
                    printf("\n\n");
                    // break;
                }
            }


        }
        printf("Speedup %f\n", (float)v1_time/v2_time);
        if(wrong_cnt) printf("Wrong %d times\n", wrong_cnt);

        free(X);
    }
    
    free(knn1.nidx);
    free(knn1.ndist);
    
    free(knn2.nidx);
    free(knn2.ndist);

    MPI_Finalize();
    return 0;
}
