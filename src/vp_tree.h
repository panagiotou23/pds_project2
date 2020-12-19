#include <stdlib.h>
#include <string.h>

typedef struct vp_tree{
    int n,
        d,
        B,
        *id,
        *left_cnt,
        *right_cnt;
    double *p,
           *mu;

}vp_tree;

/*************************FIX SELECT VP*****************************/
int select_vp(double *X, int *id, int n, int d){
    int i = rand()%n;
    return i;
}

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

    int vp = select_vp(X, id, n, vpt->d);

    vpt->id[index] = id[vp];
    memcpy(vpt->p + index * vpt->d, X + id[vp] * vpt->d, vpt->d * sizeof(double));

    double sum = 0;
    for(int i=0; i<n; i++)
        for(int j=0; j<vpt->d; j++) 
            sum += (vpt->p[j + index * vpt->d] - X[j + id[i] * vpt->d]) * 
                    (vpt->p[j + index * vpt->d] - X[j + id[i] * vpt->d]);

    vpt->mu[index] = sum/(n-1);

    int left_cnt = 0,
        right_cnt = 0,
        *left_id = malloc(n * sizeof(int)),
        *right_id = malloc(n * sizeof(int));

    for(int i=0; i<n; i++){
        double dst = 0;
        for(int j=0; j<vpt->d; j++) 
            dst += (vpt->p[j + index * vpt->d] - X[j + id[i] * vpt->d]) *
                    (vpt->p[j + index * vpt->d] - X[j + id[i] * vpt->d]);
        
        if(dst == 0) continue;

        if(dst < vpt->mu[index]){
            left_id[left_cnt++] = id[i];
        }else{
            right_id[right_cnt++] = id[i];
        }
    }

    vpt->left_cnt[index] = left_cnt;
    vpt->right_cnt[index] = right_cnt;

    make_vp_node(X, left_id, vpt, index + 1, left_cnt);
    make_vp_node(X, right_id, vpt, index + left_cnt+1, right_cnt);

}

vp_tree make_vp_tree(double *X, int n, int d, int B){
    vp_tree vpt;
    vpt.n = n;
    vpt.d = d;
    
    vpt.id = malloc(n * sizeof(int));
    vpt.p = malloc(n * d * sizeof(double));
    vpt.mu = malloc(n * sizeof(double));

    vpt.left_cnt = malloc(n * sizeof(int));
    vpt.right_cnt = malloc(n * sizeof(int));

    vpt.B = B;

    int *id = malloc(n * sizeof(int));

    for(int i=0; i<n; i++) id[i] = i;

    make_vp_node(X, id, &vpt, 0, n);

    return vpt;
}


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

void search(vp_tree vpt, int *nidx, double *ndist, int k, double *q, int index, int isLeaf, int points);

void search_l(vp_tree vpt, int *nidx, double *ndist, int k, double *q, int index){
    if(vpt.left_cnt[index] > vpt.B){
        search(vpt, nidx, ndist, k, q, index + 1, 0, 1);    
    }else{
        search(vpt, nidx, ndist, k, q, index + 1, 1, vpt.left_cnt[index]);    
    }
}

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

    if(x < ndist[k - 1]){
        add(nidx, ndist, k, vpt.id[index], x);
    }

    if(!isLeaf){
        if(x < vpt.mu[index] - ndist[k - 1]){
            search_l(vpt, nidx, ndist, k, q, index);
        }else if(x > vpt.mu[index] + ndist[k - 1]){
            search_r(vpt, nidx, ndist, k, q, index);
        }else{
            search_l(vpt, nidx, ndist, k, q, index);
            search_r(vpt, nidx, ndist, k, q, index);
        }
    }else{
        for(int i=1; i<points; i++){
            x = 0;
            for(int j=0; j<vpt.d; j++) 
                x += (vpt.p[j + (index + i) * vpt.d] - q[j]) *
                        (vpt.p[j + (index + i) * vpt.d] - q[j]);
        
            if(x < ndist[k - 1]){
                add(nidx, ndist, k, vpt.id[index + i], x);
            }
        }
    }
    
}
