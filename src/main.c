#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include "v1.h"

int main(int argc, char *argv[]){
    
    //Initialze rand()
    srand(time(0));

    //Set the number and dimensions of the data points
    int n = 20000;
    int d = 3;
    //Set the number of nearest neighbours
    int k = 10;

    //Initialize the corpus data points
    double *X = (double *)malloc(n * d * sizeof(double));
    for(int i=0; i<n*d; i++) X[i] = (double)(rand()%100000)/1000;

    //Declare the variables used to time the function
    struct timespec ts_start;
    struct timespec ts_end;

    //Start the clock
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    //Find the knn of the corpus data points
    knnresult knn = distrAllkNN(X, n, d, k);

    //Stop the clock
    clock_gettime(CLOCK_MONOTONIC, &ts_end);

    //If the return struct is not empty
    if(knn.k != 0){
        //Print the knn and their indices
        for(int i=0; i<knn.k; i++){
            printf("Dist: %lf  \tIdx: %d\n", knn.ndist[i], knn.nidx[i]);
        }
        printf("\n\n");

        //Print the time of V1
        long time = (ts_end.tv_sec - ts_start.tv_sec)* 1000000 + (ts_end.tv_nsec - ts_start.tv_nsec)/ 1000;
        printf("Time \n%ld us\n%f s\n", time, (float)time*1e-6);     
    }   

    //Free the corpus data points
    free(X);
    return 0;
}