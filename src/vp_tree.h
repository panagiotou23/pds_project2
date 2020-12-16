#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

#include "v1.h"

typedef struct node {
    int *id,
        n;
    double *p,
           mu;
    struct node *left,
                *right;
    
}node;

node select_vp(double *X, int *id, int n, int d){
    int i = rand()%n;

    node nd;
    nd.n = 1;
    nd.p = malloc(d * sizeof(double));
    memcpy(nd.p, X + i*d, d * sizeof(double));
    nd.id = malloc(sizeof(int));
    nd.id[0] = id[i];

    return nd;
}

node make_vp_tree(double *X, int *id, int n, int d, int k){

    node nd;

    if(n<=k){
        // printf("%d\n", n);
        nd.n = n;
        nd.p = malloc(n * d * sizeof(double));
        memcpy(nd.p, X, n * d * sizeof(double));

        nd.id = malloc(n * sizeof(int));
        memcpy(nd.id, id, n * sizeof(int));

        nd.left = NULL;
        nd.right = NULL;

        return nd;
    }

    nd = select_vp(X, id, n, d);

    double sum = 0;
    for(int i=0; i<n; i++)
        for(int j=0; j<d; j++) 
            sum += (nd.p[j] - X[j + i*d]) * (nd.p[j] - X[j + i*d]);
    
    nd.mu = sum / (n-1);

    double dst;

    int left_cnt = 0,
        right_cnt = 0;
    // printf("AAAAAAAAAAA\n");
    int *left_idx = malloc(n * sizeof(int));
    int *right_idx = malloc(n * sizeof(int));

    // printf("AAAAAAAAAAA\n");    
    int *left_id = malloc(n * sizeof(int));
    int *right_id = malloc(n * sizeof(int));

    for(int i=0; i<n; i++){
        dst = 0;
        for(int j=0; j<d; j++)
            dst += (nd.p[j] - X[j + i*d]) * (nd.p[j] - X[j + i*d]);

        if(dst == 0) continue;

        if(dst < nd.mu){
            left_id[left_cnt] = id[i];
            left_idx[left_cnt++] = i;
        }else{
            right_id[right_cnt] = id[i];
            right_idx[right_cnt++] = i;
        }

    }
    double *left = malloc(left_cnt * d * sizeof(double));
    double *right = malloc(right_cnt * d * sizeof(double));

    for(int i=0; i<left_cnt; i++) memcpy(left + i * d, X + left_idx[i] * d, d * sizeof(double));
    for(int i=0; i<right_cnt; i++) memcpy(right + i * d, X + right_idx[i] * d, d * sizeof(double));
    
    nd.left = (node *)malloc(sizeof(node));
    nd.right = (node *)malloc(sizeof(node));
    if(left_cnt > 0){
        *nd.left = make_vp_tree(left, left_id, left_cnt, d, k);
    }else{
        nd.left = NULL;
    }if(right_cnt > 0){
        *nd.right = make_vp_tree(right, right_id, right_cnt, d, k);
    }else{
        nd.right = NULL;
    }
    return nd;

}

void add(node nd, int idx, knnresult *knn, double x){
    for(int i=0; i<knn->k; i++){
        if(knn->ndist[i] >= x){
            memmove(knn->ndist + i + 1, knn->ndist + i, (knn->k - i - 1) * sizeof(double));
            memmove(knn->nidx + i + 1, knn->nidx + i, (knn->k - i - 1) * sizeof(double));
            knn->ndist[i] = x;
            knn->nidx[i] = nd.id[idx];
            break;
        }
    }
}

void search(node nd, knnresult *knn, double *Y, int d, double prev){

    double x;

    for(int i=0; i<nd.n; i++){
        x = 0;
        
        for(int j=0; j<d; j++) x += (nd.p[j + i * d] - Y[j]) * (nd.p[j + i * d] - Y[j]);

        if(x == 0) continue;

        if(x < knn->ndist[knn->k - 1]){
            add(nd, i, knn, x);
            prev = x;
        }
    }

    if(nd.n > 1) return;

    if(x < nd.mu - prev && prev != INFINITY){
        if(nd.left != NULL) search(*nd.left, knn, Y, d, prev);
    }else if(x > nd.mu + prev && prev != INFINITY){
        if(nd.right != NULL) search(*nd.right, knn, Y, d, prev);
    }else{
        if(nd.left != NULL) search(*nd.left, knn, Y, d, prev);
        if(nd.right != NULL) search(*nd.right, knn, Y, d, prev);
    }
}