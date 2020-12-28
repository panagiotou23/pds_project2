#include <string.h>
#include <math.h>

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

int partition(int * nidx,   
            double * ndist,
            int left,
            int right,
            int pivot)
{

    double val = ndist[pivot];

    SWAP(ndist[right], ndist[pivot], double);
    SWAP(nidx[right], nidx[pivot], int);
    
    int storeIdx = left;

    for(int i=left; i<right; i++){
        if(ndist[i] < val){
            SWAP(ndist[storeIdx], ndist[i], double);
            SWAP(nidx[storeIdx], nidx[i], int);
            storeIdx++;

        }
    }
    SWAP(ndist[storeIdx], ndist[right], double);
    SWAP(nidx[storeIdx], nidx[right], int);
    return storeIdx;
}

void quickselect(int * nidx,   
            double * ndist,
            int left,
            int right,
            int k)
{
    if(left >= right) return;
    
    int pivot = left + rand() % (right - left + 1);
    pivot = partition(nidx, ndist, left, right, pivot);

    if(k == pivot){
        return;
    }else if(k< pivot){
        quickselect(nidx, ndist, left, pivot - 1, k);
    }else{
        quickselect(nidx, ndist, pivot + 1, right, k);
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

void calc_part_knn( double *D,
                    int n,
                    int m,
                    int k,
                    int col,
                    int *final_nidx,
                    double *final_ndist)
{

    int * nidx = (int *)malloc(n * sizeof(int));
    double * ndist = (double *)malloc(n * sizeof(double));

    for(int i=0; i<n; i++){
        nidx[i] = i;
        ndist[i] = D[i * m + col];
    }

    quickselect(nidx, ndist, 0, n-1, k);
    
    memcpy(final_nidx, nidx, k * sizeof(int));
    memcpy(final_ndist, ndist, k * sizeof(double));
    
    free(nidx);
    free(ndist);
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

    int max_m = 1e3;
    if(m > max_m){
        
        int iteration_m[m/max_m];
        for(int i=0; i<m/max_m; i++) iteration_m[i] = max_m;
        iteration_m[m/max_m - 1] = max_m + m%max_m;
        
        int displ = 0;
        for(int i=0; i<m/max_m; i++){
            double *D = calc_D(X, Y + displ * d, n, d, iteration_m[i]);

            for(int j=0; j<iteration_m[i]; j++) calc_part_knn(D, n, iteration_m[i], k, j, knn.nidx + (j + displ) * k, knn.ndist + (j + displ) * k);

            free(D);
            displ += iteration_m[i];
        }

    }else{

        double *D = calc_D(X, Y, n, d, m);

        for(int j=0; j<m; j++) calc_part_knn(D, n, m, k, j, knn.nidx + j * k, knn.ndist + j * k);

        free(D);

    }

    return knn;
}

knnresult alt_kNN(  double *X,
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

    for(int j=0; j<m; j++){

        double * ndist = calc_Dcol(X, Y, n, d, j);

        int * nidx = (int *)malloc(n * sizeof(int));
        for(int i=0; i<n; i++) nidx[i] = i;

        quickselect(nidx, ndist, 0, n-1, k);
        
        memcpy(knn.nidx + j * k, nidx, k * sizeof(int));
        memcpy(knn.ndist + j * k, ndist, k * sizeof(double));
        
        free(nidx);
        free(ndist);
    }
    
    return knn;
}

