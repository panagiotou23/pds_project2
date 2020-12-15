#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include "v1.h"
#include "v2.h"
// #include "v2.1.h"
// #include "vp_tree.h"

int main(int argc, char *argv[]){
    
    srand(time(0));

    int n = 100;
    int d = 2;
    int k = 10;


    double *X = (double *)malloc(n * d * sizeof(double));
    int *idx = (int *)malloc(n * sizeof(int));

    for(int i=0; i<n*d; i++) {
        X[i] = (double)(rand()%100000)/1000;
        idx[i] = i;
    }

    // for(int i=0; i<n; i++){
    //     for(int j=0; j<d; j++) printf("%lf ", X[j + i*d]);
    //     printf("\n");
    // }

    // vps_node nod = make_vps_tree(X, n, d);

    node nod = make_vp_tree(X, idx, n, d);

    // node nd = make_vp_tree(X, n, d, k);
    
    // for(int i=0; i<n; i++){

    //     int id = 0;
    //     double min = INFINITY;
    //     search(nod, X + i*d, d, &min, &id);

    //     printf("min: %lf\tid: %d\n",min, id);

    // }
    return 0;

    //The Variables used to time the function
    struct timespec ts_start;
    struct timespec ts_end;

    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    knnresult knn = distrAllkNN(X, n, d, k);
    clock_gettime(CLOCK_MONOTONIC, &ts_end);

    if(knn.k != 0){
        for(int i=0; i<knn.k; i++){
            printf("Dist: %lf  \tIdx: %d\n", knn.ndist[i], knn.nidx[i]);
        }
        printf("\n\n");

        long time = (ts_end.tv_sec - ts_start.tv_sec)* 1000000 + (ts_end.tv_nsec - ts_start.tv_nsec)/ 1000;
        printf("Time \n%ld us\n%f s\n", time, (float)time*1e-6);     
    }   

    free(X);
    return 0;
}