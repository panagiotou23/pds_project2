#include <stdio.h>
#include <time.h>

#include "v2.h"
#include "read_X.h"

double *create_X(int n, int d){

    //Initialize the corpus data points
    double *X = (double *)malloc(n * d * sizeof(double));
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

    double *X;
    int n,
        d,
        k;
    
    if(strstr(argv[1],"-c") != NULL){
        n = atof(argv[2]) * 1e3;
        d = atoi(argv[3]);
        k = atoi(argv[4]);

        if(!world_rank) {
            X = create_X(n, d);
        }
        
    }else if(argv[1],"-r"){
        if(!world_rank){
            X = read_X(&n, &d, argv[2]);
        }
        k = atoi(argv[3]);

        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&d, 1, MPI_INT, 0, MPI_COMM_WORLD);

    }else{
        printf("Wrong input\n\n\tUsage\n");
        printf("%s -c n d k\n", argv[0]);
        printf("\t Or\n%s -r path/to/file k\n", argv[0]);
        return -1;
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
        // printf("V1 time\n%ld us\n%f s\n\n", v1_time, v1_time*1e-6);
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
        // printf("V2 time\n%ld us\n%f s\n\n", v2_time, v2_time*1e-6);
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
                    // printf("%d\nDiff: %lf\n", i, knn1.ndist[j + i * k] - knn2.ndist[j + i * k]);
                    // printf("Dist: %lf  \tIdx: %d\n", knn1.ndist[j + i * k], knn1.nidx[j + i * k]);
                    // printf("Dist: %lf  \tIdx: %d\n", knn2.ndist[j + i * k], knn2.nidx[j + i * k]);
                    // printf("\n\n");
                    // break;
                }
            }


        }
        printf("Speedup %f\n\n\n", (float)v1_time/v2_time);
        // if(wrong_cnt) printf("Wrong %d times\n", wrong_cnt);

        free(X);
    }
    
    free(knn1.nidx);
    free(knn1.ndist);
    
    free(knn2.nidx);
    free(knn2.ndist);

    MPI_Finalize();
    return 0;
}
