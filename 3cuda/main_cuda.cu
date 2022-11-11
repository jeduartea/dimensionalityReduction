/*******************************************************************************
*
* Program: Dimensionality Reduction With Entropy to a data Matrix set
* 
* Description: Project - Distributed Computing
* 
* Universidad Nacional de Colombia
* 
* Author: Javier Eduardo Duarte Aunta
*
* Octuber 2022
*******************************************************************************/

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#define MATRIX_ROW 2560
#define MATRIX_COLUNM 10
#define STRING_LENGTH 4
#define MAX_LEN_LINE STRING_LENGTH*MATRIX_COLUNM

#define PRINT_LOGS 0

#define BLOCKSPERGRID  40
#define NUMTHREADS 64


//Matrix type char (*)[MATRIX_COLUNM][STRING_LENGTH]
char main_matrix[MATRIX_ROW][MATRIX_COLUNM][STRING_LENGTH]={/*  */};

//Matrix type double (*)[MATRIX_ROW]
double similarity_matrix[MATRIX_ROW][MATRIX_ROW]={0.0};

// function to calculate Hamming distance
__device__ int hammingDist_CUDA (char* str1, char* str2)
{
    int i = 0, count = 0;
    while (str1[i] != '\0') {
        if (str1[i] == str2[i] && str1[i] != '\n' && str2[i] != '\n')
            count++;
        i++;
    }
    return count;
}


void readCSVFile (char main_matrix[MATRIX_ROW][MATRIX_COLUNM][STRING_LENGTH])
{
    
    if (PRINT_LOGS){
    printf("-----------------------------------------\n");
    printf("BEGIN - function: readCSVFile \n");
    printf("-----------------------------------------\n");
    }

   char line[MAX_LEN_LINE];
   char *token;

    FILE *file;
    file = fopen("matrix.txt", "r");


    if (file == NULL){
        printf("Error opening file.\n");
    }

    for (int row = 0; row < MATRIX_ROW; row++){
        fgets(line, MAX_LEN_LINE, file);
        //printf("%s", line);

        token= strtok(line, ",");
        for (int colunm = 0; colunm < MATRIX_COLUNM; colunm++)
        {
            if (colunm == MATRIX_COLUNM-1){
                token= strtok(token, "\n");
                strcpy(main_matrix[row][colunm],token);
            }
            else{
                strcpy(main_matrix[row][colunm],token);
                token= strtok(NULL, ",");
            }

        }
        
    }
    fclose(file);

    if (PRINT_LOGS){   
    printf("-----------------------------------------\n");
    printf("END - function: readCSVFile \n");
    printf("-----------------------------------------\n");
    }
}


__global__ void similarityMatrixCalculation_CUDA (char *data_matrix, double *similarity_matrix, int withoutColum)
{
    int index_row = (blockDim.x * blockIdx.x) + threadIdx.x;

    double sum=0;

    //for(int row_s=0; row_s < MATRIX_ROW; row_s++){

    for(int column_s=0; column_s< MATRIX_ROW; column_s++){

        for (int column_d=0; column_d < MATRIX_COLUNM; column_d++){
            if (withoutColum == 0){

                sum = sum + hammingDist_CUDA(&data_matrix[index_row*MATRIX_COLUNM*STRING_LENGTH + column_d*STRING_LENGTH], &data_matrix[column_s*MATRIX_COLUNM*STRING_LENGTH + column_d*STRING_LENGTH]);
                //sum = sum + hammingDist_CUDA(data_matrix[row_s][column_d],data_matrix[column_s][column_d]);
            }
            else{
                if (withoutColum-1 != column_d)
                    sum = sum + hammingDist_CUDA(&data_matrix[index_row*MATRIX_COLUNM*STRING_LENGTH + column_d*STRING_LENGTH], &data_matrix[column_s*MATRIX_COLUNM*STRING_LENGTH + column_d*STRING_LENGTH]);
                    //sum = sum + hammingDist_CUDA(data_matrix[row_s][column_d],data_matrix[column_s][column_d]);

            }
            
        }

        if (withoutColum == 0){
            similarity_matrix[index_row*MATRIX_ROW + column_s]= (double) sum/MATRIX_COLUNM;
            //similarity_matrix[index_row][column_s]= (double) sum/MATRIX_COLUNM;
        }


        
        else{
            similarity_matrix[index_row*MATRIX_ROW + column_s]= (double) sum/(MATRIX_COLUNM-1);
            //similarity_matrix[index_row][column_s]= (double) sum/(MATRIX_COLUNM-1);
        }
        sum=0;
    }

    ////}
}

void entropyCalculation(double similarity_matrix[MATRIX_ROW][MATRIX_ROW], double final_entropy[MATRIX_ROW], int withoutColum){

    double entropy=0;
    for (int i=0; i < MATRIX_ROW-2; i++){
        for (int j=i+1; j < MATRIX_ROW-1;j++){
            if (similarity_matrix[i][j] != 0.0)
            entropy = entropy + ((similarity_matrix[i][j]*log(similarity_matrix[i][j])) +  ((1-similarity_matrix[i][j]) * log(1-similarity_matrix[i][j])));
        }
    }
    entropy = (-1) * entropy;
    final_entropy[withoutColum] = entropy;
}

int main ()
{

    printf("****************************************************************************\n");
    printf("    BEGIN - Entropy Calculation for Dimensionality Reduction with CUDA\n");
    printf("***************************************************************************\n\n");

    // Varibles for mesuered time elapsed
    struct timeval tval_before, tval_after, tval_result;
    gettimeofday(&tval_before, NULL);

    //reading CSV file
    readCSVFile(main_matrix);

    //*******************************************************
    // BEGIN CUDA CODE
    //*******************************************************

    
    // Flating main matrix AND similarity matrix
    // varibles in host
    char *h_main_matrix = (char *) main_matrix;
    double *h_similarity_matrix = (double *) similarity_matrix;

    //Array that has all entropy values
    double h_entropy[MATRIX_COLUNM+1];

    // varibles in device
    char *d_main_matrix;
    double *d_similarity_matrix;
    double *d_entropy;

    //getting size of each varible
    int size_main_matrix = sizeof(main_matrix);
    int size_similarity_matrix = sizeof(double)*MATRIX_ROW*MATRIX_ROW;
    int size_entropy = sizeof(h_entropy);

 
    //---------------------------
    // BEGIN - ALLOCATE IN DEVICE
    //---------------------------

    cudaError_t err = cudaSuccess;

    //For main_matrix
    cudaMalloc( (void **) &d_main_matrix, size_main_matrix);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to allocate d_main_matrix in device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //For similarity_matrix
    cudaMalloc( (void **) &d_similarity_matrix, size_similarity_matrix);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to allocate d_similarity_matrix in device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //For h_entropy
    cudaMalloc( (void **) &d_entropy, size_entropy);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to allocate d_entropy in device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //---------------------------
    // END - ALLOCATE IN DEVICE
    //---------------------------


    //---------------------------
    // BEGIN - COPY VAULE TO DEVICE
    //---------------------------

    //For main_matrix
    err = cudaMemcpy(d_main_matrix, h_main_matrix, size_main_matrix, cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to copy vector h_main_matrix from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //For similarity_matrix

    //For h_entropy

    //---------------------------
    // END  - COPY VAULE TO DEVICE
    //---------------------------


    //---------------------------
    // BEGIN - LAUNCH KERNEL
    //---------------------------
    // Launch kernel N times (numbers columns) 

    for (int column = 0; column < MATRIX_COLUNM+1; column++){
    similarityMatrixCalculation_CUDA<<<NUMTHREADS,BLOCKSPERGRID>>>(d_main_matrix, d_similarity_matrix, column);

        //---------------------------
        // BEGIN - COPY VAULES TO HOST
        //---------------------------
        //h_similarity_matrix
        err = cudaMemcpy(h_similarity_matrix, d_similarity_matrix, size_similarity_matrix, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess){
            fprintf(stderr, "Failed to copy vector h_similarity_matrix from device to host (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        entropyCalculation(similarity_matrix, h_entropy, column);
        //---------------------------
        // END -  COPY VAULES TO HOST
        //---------------------------
    }
    //---------------------------
    // END -  LAUNCH KERNEL
    //---------------------------


    //---------------------------
    // BEGIN - PRINT DATA FROM GPU
    //---------------------------

    //For h_entropy
    printf("\n");
    printf("The entropy values for each column are as follows:\n");
    for (int i = 0; i < MATRIX_COLUNM+1; i++){
        if (i==0) // 0 in is saved the entropy vaule with all the columns
            printf("    The value with all columns    : %f\n",h_entropy[i]);

        else
            printf("    The value without the column %d: %f\n",i,h_entropy[i]);
    }

    //---------------------------
    // END -  PRINT DATA FROM GPU
    //---------------------------


    //---------------------------
    // BEGIN - FREE MEMORY 
    //---------------------------

    //For d_main_matrix
    err = cudaFree(d_main_matrix);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to free device vector d_main_matrix (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //For similarity_matrix
    err = cudaFree(d_similarity_matrix);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to free device vector d_similarity_matrix (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //For h_entropy
    err = cudaFree(d_entropy);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to free device vector d_entropy (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //---------------------------
    // END - FREE MEMORY
    //---------------------------

    //*******************************************************
    // END - CUDA CODE
    //*******************************************************

    // getting Time elapsed
    gettimeofday(&tval_after, NULL);
    timersub(&tval_after, &tval_before, &tval_result);
    
    printf("\n****************************************************************************\n");
    printf("    END - Entropy Calculation for Dimensionality Reduction with CUDA\n");
    printf("***************************************************************************\n");
    printf("\nTime elapsed: %ld.%06ld\n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
    return 0;
}
