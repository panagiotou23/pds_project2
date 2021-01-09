#include <stdio.h>
#include <stdlib.h>
#include <string.h>

double *read_corel(int *n, int *d, char *file_path){

    FILE *matFile;
    matFile = fopen(file_path, "r");
    if (matFile == NULL){
        printf("Could not open file %s",file_path);
        exit(-1);
    }

    double *X;
    if(strstr(file_path,"ColorH") != NULL){
        *n = 68040;
        *d = 32;
    }else if(strstr(file_path,"ColorM") != NULL){
        *n = 68040;
        *d = 9;
    }else{
        *n = 68040;
        *d = 16;
    }

    X = (double *)malloc((*n) * (*d) * sizeof(double));

    for(int i=0; i<*n; i++){
        int row;
        int got = fscanf(matFile, "%d", &row);
        for(int j=0; j<*d; j++){
            int got = fscanf(matFile, "%lf", &X[i * (*d) + j]);
            if(got != 1){
                printf("Error reading\n");
                exit(-2);
            }
        }
    }

    fclose(matFile);

    return X;
}

double *read_mini(int *n, int *d, char *file_path){

    FILE *matFile;
    matFile = fopen(file_path, "r");
    if (matFile == NULL){
        printf("Could not open file %s",file_path);
        exit(-1);
    }
    
    double *X;
    *n = 130064;
    *d = 50;

    X = (double *)malloc((*n) * (*d) * sizeof(double));
    
    int temp;
    for(int i=0; i<2; i++){
        int got = fscanf(matFile, "%d", &temp);
    }

    for(int i=0; i<*n; i++){
        for(int j=0; j<*d; j++){
            int got = fscanf(matFile, "%lf", &X[i * (*d) + j]);
            if(got != 1){
                printf("Error reading\n");
                exit(-2);
            }
        }
    }

    fclose(matFile);

    return X;
}

double *read_features(int *n, int *d, char *file_path){

    FILE *matFile;
    matFile = fopen(file_path, "r");
    if (matFile == NULL){
        printf("Could not open file %s",file_path);
        exit(-1);
    }
    
    char *line = (char *)malloc(1024 * 1024);

    for(int skip=0;skip<4;skip++){
        int got = fscanf(matFile,"%s\n", line);
    }
    free(line);

    double *X;
    *n = 106574;
    *d = 518;

    X = (double *)malloc((*n) * (*d) * sizeof(double));

    int temp;    
    for(int i=0; i<*n; i++){
        int got = fscanf(matFile,"%d,",&temp);
        for(int j=0; j<*d; j++){
            int got = fscanf(matFile,"%lf,",&X[i * (*d) + j]);
            if(got != 1){
                printf("Error reading\n");
                exit(-2);
            }
        }
    }
 
    fclose(matFile);
    return X;
}

double *read_tv(int *n, int *d, char *file_path){

    FILE *matFile;
    matFile = fopen(file_path, "r");
    if (matFile == NULL){
        printf("Could not open file %s",file_path);
        exit(-1);
    }
    
    double *X;
    if(strstr(file_path,"BBC") != NULL){
        *n = 17720;
    }else if(strstr(file_path,"CNN.") != NULL){
        *n = 22545;
    }else if(strstr(file_path,"CNNI") != NULL){
        *n = 33117;
    }else if(strstr(file_path,"NDTV") != NULL){
        *n = 17051;
    }else if(strstr(file_path,"TIMES") != NULL){
        *n = 39252;
    }
    *d = 17;

    X = (double *)malloc((*n) * (*d) * sizeof(double));

    int temp;
    double val;
    int got = fscanf(matFile, "%d", &temp);
    for(int i=0; i<*n; i++){
        for(int j=0; j<*d; j++){
            int got = fscanf(matFile, "%d:%lf", &temp, &val);
            if(got != 2){
                break;
            }
            X[i * (*d) + j] = val;
        }
        while(1){
            int got = fscanf(matFile, "%d:%lf", &temp, &val);
            if(got != 2){
                break;
            }
        }
    }

    fclose(matFile);

    return X;
}

double *read_X(int *n, int *d, char *file_path){
    double *X;
    
    if(strstr(file_path,"Co") != NULL){
        X = read_corel(n, d, file_path);
    }else if(strstr(file_path,"Mini") != NULL){
        X = read_mini(n, d, file_path);
    }else if(strstr(file_path,"feat") != NULL){
        X = read_features(n, d, file_path);
    }else if(strstr(file_path,"BBC") != NULL ||
            strstr(file_path,"CNN.") != NULL ||
            strstr(file_path,"CNNI") != NULL ||
            strstr(file_path,"NDTV") != NULL ||
            strstr(file_path,"TIMES") != NULL){
        X = read_tv(n, d, file_path);
    }else{
        printf("Don't know how to read this File\n");
        exit(2);
    }
    
    return X;
}