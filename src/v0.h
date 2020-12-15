#include <stdbool.h>
#include <string.h>

#include <cblas.h>

#define SWAP(x, y, T) do { T SWAP = x; x = y; y = SWAP; } while (0)

//The Struct that contains the k nearest neighbors of m queries
typedef struct knnresult{
  int    * nidx;    //!< Indices (0-based) of nearest neighbors [m-by-k]
  double * ndist;   //!< Distance of nearest neighbors          [m-by-k]
  int      m;       //!< Number of query points                 [scalar]
  int      k;       //!< Number of nearest neighbors            [scalar]
} knnresult;

//A sorting alogorithm for the k first items of a list
void k_select(int * nidx,   
            double * ndist,
            int k,
            int n)
{
    for(int i=0; i<k; i++){
        int minidx = i;
        double min = ndist[i];
        for(int j=i+1; j<n; j++){
            if(min > ndist[j]){
                min = ndist[j];
                minidx = nidx[j];
                SWAP(ndist[i],ndist[j], double);
                SWAP(nidx[i], nidx[j], int);
            }
        } 
    }
}

//Calculates a column of the Euclidean distance matrix D 
double *calc_Dcol(  double *X,
                    double *Y,
                    int n,
                    int d,
                    int row)
{
    double *Dcol= (double *)malloc(n * sizeof(double));

    for(int i=0; i<n; i++) Dcol[i] = 0;

    for(int i=0; i<n; i++){
        for(int k=0; k<d; k++){
            Dcol[i] += X[k + i*d]*X[k + i*d] + Y[k + row*d]*Y[k + row*d] - 2*X[k + i*d]*Y[k + row*d];
        }
    }

    return Dcol;
}

//Calculates the Euclidean distance matrix D 
double *calc_D( double *X,
                double *Y,
                int n,
                int d,
                int m)
{
    double *D= (double *)malloc(n * m * sizeof(double));
    
    double xsum[n],
           ysum[m];
    for(int i=0; i<n; i++) xsum[i] = cblas_ddot(d, X + i*d, 1, X + i*d, 1);
    for(int i=0; i<m; i++) ysum[i] = cblas_ddot(d, Y + i*d, 1, Y + i*d, 1);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, m, d, -2, X, d, Y, d, 0, D, m);
    
    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){
            D[j + m*i] += xsum[i] + ysum[j]; 
        }
    }

    return D;
}

//Finds for each point in a query set Y the k nearest neighbors
//in the corpus set X
knnresult kNN(  double *X,
                double *Y,
                int n,
                int m,
                int d,
                int k)
{

    knnresult knn;
    knn.m = m;
    knn.k = k;
    knn.nidx = (int *)malloc(m * k * sizeof(int));
    knn.ndist = (double *)malloc(m * k * sizeof(double));

    //For every point in Y
    for(int j=0; j<m; j++){

        //Calcute its distance from every point in X 
        double * ndist = calc_Dcol(X, Y, n, d, j);

        //And memorize their indices
        int * nidx = (int *)malloc(n * sizeof(int));
        for(int i=0; i<n; i++) nidx[i] = i;
        
        //Sort the k + 1 distances and indices,
        //because it will also calculate the distance from itself
        k_select(nidx,ndist, k + 1, n);

        //Save only the k smallest distances except from itself
        memcpy(knn.nidx + j * k, nidx + 1, k * sizeof(int));
        memcpy(knn.ndist + j * k, ndist + 1, k * sizeof(double));
        
        //Free the arrays
        free(nidx);
        free(ndist);
    }
    
    return knn;
}

