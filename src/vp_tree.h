#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct vp_tree{
    int n,              //total points of the tree
        d,              //dimension of the points
        B,              //max points per lef
        *id,            //id of every point in the tree
        *left_cnt,      //number of left children of every point in the tree 
        *right_cnt;     //number of right children of every point in the tree
    double *p,          //points of the tree
           *mu;         //median distance of each point from its subset

}vp_tree;

//Selcets the vantage point from a specified subset of points
int select_vp(double *X, int *id, int n, int d, int max){

    int *indices = (int *)malloc(max * sizeof(int)),
        *sample = (int *)malloc(max * sizeof(int));
    for(int i=0; i<max; i++) {
        sample[i] = rand()%n;
        indices[i] = rand()%n;
    }
    
    double best_spread = 0;
    int best_i;

    double sum = 0; 
    for(int k=0; k<max; k++){

        for(int i=0; i<max; i++){
            for(int j=0; j<d; j++){
                sum += (X[j + indices[k] * d] - X[j + sample[i] * d]) *
                        (X[j + indices[k] * d] - X[j + sample[i] * d]);
            }
        }

        double mu = sqrt(sum)/max;
        double spread = 0;
        for(int i=0; i<max; i++){
            for(int j=0; j<d; j++){
                spread += (X[j + sample[i] * d] - mu) *
                            (X[j + sample[i] * d] - mu);
            }
        }
        spread /= (n-1);

        if(spread > best_spread){
            best_spread = spread;
            best_i = indices[k];
        }

    }   

    free(sample);
    free(indices);

    return best_i;
}

//Add a node to the vantge point tree
void make_vp_node(double *X, int *id, vp_tree *vpt, int index, int n){
    
    if(n <= vpt->B){
        for(int i=0; i<n; i++){
            vpt->id[index + i] = id[i];
            memcpy(vpt->p + (index + i) * vpt->d, X + id[i] * vpt->d, vpt->d * sizeof(double));
            vpt->mu[index + i] = 0;
            vpt->left_cnt[index+i] = 0;
            vpt->right_cnt[index+i] = 0;
        }
        return;
    }

    int vp = select_vp(X, id, n, vpt->d, vpt->B);
    vpt->id[index] = id[vp];
    memcpy(vpt->p + index * vpt->d, X + id[vp] * vpt->d, vpt->d * sizeof(double));

    double *distances = (double *)malloc(n * sizeof(double));
    double sum = 0;

    for(int i=0; i<n; i++){
        distances[i] = 0;
        for(int j=0; j<vpt->d; j++){
            distances[i] += (vpt->p[j + index * vpt->d] - X[j + id[i] * vpt->d]) *
                    (vpt->p[j + index * vpt->d] - X[j + id[i] * vpt->d]);
        }
        sum += distances[i];
        distances[i] = sqrt(distances[i]);
    }

    vpt->mu[index] = sqrt(sum)/(n-1);

    int left_cnt = 0,
        right_cnt = 0,
        *left_id = (int *)malloc(n * sizeof(int)),
        *right_id = (int *)malloc(n * sizeof(int));

    for(int i=0; i<n; i++){
        if(distances[i] == 0) continue;

        if(distances[i] < vpt->mu[index]){
            left_id[left_cnt++] = id[i];
        }else{
            right_id[right_cnt++] = id[i];
        }
    }
    free(distances);

    vpt->left_cnt[index] = left_cnt;
    vpt->right_cnt[index] = right_cnt;

    make_vp_node(X, left_id, vpt, index + 1, left_cnt);
    free(left_id);
    make_vp_node(X, right_id, vpt, index + left_cnt+1, right_cnt);
    free(right_id);

}

//Creates the vantage point tree
vp_tree make_vp_tree(double *X, int n, int d, int B){
    vp_tree vpt;
    vpt.n = n;
    vpt.d = d;
    
    vpt.id = (int *)malloc(n * sizeof(int));
    vpt.p = (double *)malloc(n * d * sizeof(double));
    vpt.mu = (double *)malloc(n * sizeof(double));

    vpt.left_cnt = (int *)malloc(n * sizeof(int));
    vpt.right_cnt = (int *)malloc(n * sizeof(int));

    vpt.B = B;

    int *id = (int *)malloc(n * sizeof(int));

    for(int i=0; i<n; i++) id[i] = i;

    make_vp_node(X, id, &vpt, 0, n);
    
    free(id);
    
    return vpt;
}

//Adds the distance and the index of a point in a sorted array
void add(int *nidx, double *ndist, int k, int idx, double x){
    for(int i=0; i<k; i++){
        if(ndist[i] >= x){
            memmove(nidx + i + 1, nidx + i, (k - i - 1) * sizeof(int));
            memmove(ndist + i + 1, ndist + i, (k - i - 1) * sizeof(double));
            
            nidx[i] = idx;
            ndist[i] = x;
            break;
        }
    }
}

//Calculates the kNN of a query point in a vantage point tree
void search(vp_tree vpt, int *nidx, double *ndist, int k, double *q, int index, int isLeaf, int points);

//Searches the left subtree of a vantage point tree
void search_l(vp_tree vpt, int *nidx, double *ndist, int k, double *q, int index){
    if(vpt.left_cnt[index] > vpt.B){
        search(vpt, nidx, ndist, k, q, index + 1, 0, 1);    
    }else{
        search(vpt, nidx, ndist, k, q, index + 1, 1, vpt.left_cnt[index]);    
    }
}

//Searches the right subtree of a vantage point tree
void search_r(vp_tree vpt, int *nidx, double *ndist, int k, double *q, int index){
    if(vpt.right_cnt[index] > vpt.B){   
        search(vpt, nidx, ndist, k, q, index + vpt.left_cnt[index] + 1, 0, 1);
    }else{
        search(vpt, nidx, ndist, k, q, index + vpt.left_cnt[index] + 1, 1, vpt.right_cnt[index]);
    }
}

void search(vp_tree vpt, int *nidx, double *ndist, int k, double *q, int index, int isLeaf, int points){

    double x = 0;
    for(int j=0; j<vpt.d; j++) 
        x += (vpt.p[j + index * vpt.d] - q[j]) *
                (vpt.p[j + index * vpt.d] - q[j]);
    x = sqrt(x);
    if(x < ndist[k - 1]){
        add(nidx, ndist, k, vpt.id[index], x);
    }
    if(!isLeaf){
        if(x < vpt.mu[index] - ndist[k-1]){
            search_l(vpt, nidx, ndist, k, q, index);
        }else if(x > vpt.mu[index] + ndist[k-1]){
            search_r(vpt, nidx, ndist, k, q, index);
        }else{
            if(vpt.left_cnt[index]) search_l(vpt, nidx, ndist, k, q, index);
            if(vpt.right_cnt[index]) search_r(vpt, nidx, ndist, k, q, index);
        }
    }else{
        for(int i=1; i<points; i++){
            x = 0;
            for(int j=0; j<vpt.d; j++) 
                x += (vpt.p[j + (index + i) * vpt.d] - q[j]) *
                        (vpt.p[j + (index + i) * vpt.d] - q[j]);
            x = sqrt(x);
            if(x < ndist[k - 1]){
                add(nidx, ndist, k, vpt.id[index + i], x);
            }
        }
    }
    
}
