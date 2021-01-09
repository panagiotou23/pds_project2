#include <stdlib.h>

double *create_X(int n, int d){

    //Initialize the corpus data points
    double *X = (double *)malloc(n * d * sizeof(double));
    for(int i=0; i<n*d; i++) X[i] = (double)(rand()%100000)/1000;
    
    return X;
}
