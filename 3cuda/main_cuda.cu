#@title hello_world.cu
%%writefile hello_worldC.cu
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

#define MATRIX_ROW 1000
#define MATRIX_COLUNM 6
#define STRING_LENGTH 4
#define MAX_LEN_LINE STRING_LENGTH*MATRIX_COLUNM

#define PRINT_LOGS 0
#define PRINT_LOGS_MATRIX 0

#define BLOCKSPERGRID  1
#define NUMTHREADS 1


//Matrix type double (*)[MATRIX_ROW]
double similarity_matrix[MATRIX_ROW][MATRIX_ROW]={0.0};

//Matrix type char (*)[MATRIX_COLUNM][STRING_LENGTH]
char main_matrix[MATRIX_ROW][MATRIX_COLUNM][STRING_LENGTH]={/*  */};


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

int hammingDist(char* str1, char* str2)
{
    int i = 0, count = 0;
    while (str1[i] != '\0') {
        if (str1[i] == str2[i] && str1[i] != '\n' && str2[i] != '\n')
            count++;
        i++;
    }
    return count;
}

void printDataMatrix(char data_matrix[][MATRIX_COLUNM][STRING_LENGTH])
{
    // Print read data
    printf("Printing read matrix:\n");
    for(int i=0; i<MATRIX_ROW; i++){
        for (int j=0; j<MATRIX_COLUNM; j++){
            printf("%s ", data_matrix[i][j]);
        }
        //printf(" %d",i);
        printf("\n");
    }
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
    if (PRINT_LOGS_MATRIX)
        printDataMatrix(main_matrix);

    if (PRINT_LOGS){   
    printf("-----------------------------------------\n");
    printf("END - function: readCSVFile \n");
    printf("-----------------------------------------\n");
    }
}

void printSimilarityMatrix( double similarity_matrix[MATRIX_ROW][MATRIX_ROW])
{

    // Print similarity matrix
    printf("Printing similarity matrix:\n");
    for(int i=0; i<MATRIX_ROW; i++){
        for (int j=0; j<MATRIX_ROW; j++){
            printf("%f ", similarity_matrix[i][j]);
        }
        printf("\n");
    }

}


void entropyCalculation (char data_matrix[][MATRIX_COLUNM][STRING_LENGTH],double similarity_matrix[][MATRIX_ROW],int withoutColum, double *final_entropy)
{
    
    if (PRINT_LOGS){
    printf("-----------------------------------------\n");
    printf("BEGIN - function: entropyCalculation for column %d\n", withoutColum);
    printf("-----------------------------------------\n");
    }

    //Matrix type double (*)[MATRIX_ROW]
    //double similarity_matrix[MATRIX_ROW][MATRIX_ROW];

    if (PRINT_LOGS){
    printf("Begin - calculation of the similarity matrix for column %d ...\n\n", withoutColum);
    }

    double sum=0;
    for(int row_s=0; row_s < MATRIX_ROW; row_s++){
        for(int column_s=0; column_s< MATRIX_ROW; column_s++){

            for (int column_d=0; column_d < MATRIX_COLUNM; column_d++){
                if (withoutColum == 0){
                   sum = sum + hammingDist(data_matrix[row_s][column_d],data_matrix[column_s][column_d]);
                }
                else{
                    if (withoutColum-1 != column_d)
                        sum = sum + hammingDist(data_matrix[row_s][column_d],data_matrix[column_s][column_d]);

                }
                
            }
            if (withoutColum == 0)
                similarity_matrix[row_s][column_s]= (double) sum/MATRIX_COLUNM;
            
            else
                similarity_matrix[row_s][column_s]= (double) sum/(MATRIX_COLUNM-1);
            

            sum=0;
        }
    }

    if (PRINT_LOGS_MATRIX){
        printSimilarityMatrix(similarity_matrix);
    }
    if (PRINT_LOGS){
        printf("\nEnd - calculation of the similarity matrix for column %d.\n\n", withoutColum);
        printf("Begin - calculation of entroy for column %d...\n\n", withoutColum);
        }

    double entropy=0;
    for (int i=0; i < MATRIX_ROW-2; i++){
        for (int j=i+1; j < MATRIX_ROW-1;j++){
            if (similarity_matrix[i][j] != 0.0)
               entropy = entropy + ((similarity_matrix[i][j]*log(similarity_matrix[i][j])) +  ((1-similarity_matrix[i][j]) * log(1-similarity_matrix[i][j])));
        }
    }
    entropy = (-1)* entropy;
    if (PRINT_LOGS){
    printf("The entropy value for column %d is: %f\n\n", withoutColum,entropy);
    printf("end - calculation of entroy for column %d.\n", withoutColum);

    printf("-----------------------------------------\n");
    printf("END - function: entropyCalculation for column %d\n", withoutColum);
    printf("-----------------------------------------\n");
    }
    final_entropy[withoutColum] = entropy;
}


__global__ void entropyCalculation_CUDA (char *data_matrix, double *similarity_matrix, double *final_entropy,int withoutColum)
{
    int index_row = (blockDim.x * blockIdx.x) + threadIdx.x;

    // similarityMatrix
    double sum=0;

    // se esta rompiendo las filas de la matrix de similaridad
    // importante el numero de hilo lanzado debe ser igual al numero de filas en el data set

    //for(int row_s=0; row_s < MATRIX_ROW; row_s++){

    for(int column_s=0; column_s< MATRIX_ROW; column_s++){

        for (int column_d=0; column_d < MATRIX_COLUNM; column_d++){
            if (withoutColum == 0){
                //matrix[index_row*MATRIX_COLUNM*STRING_LENGTH + column*STRING_LENGTH + length ]
                sum = sum + hammingDist_CUDA((char *) data_matrix[index_row*MATRIX_COLUNM + column_d], (char *) data_matrix[column_s*MATRIX_COLUNM + column_d]);
                //sum = sum + hammingDist_CUDA(data_matrix[row_s][column_d],data_matrix[column_s][column_d]);
            }
            else{
                if (withoutColum-1 != column_d)
                    sum = sum + hammingDist_CUDA((char *) data_matrix[index_row*MATRIX_COLUNM + column_d],(char *) data_matrix[column_s*MATRIX_COLUNM + column_d]);
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

    __syncthreads();
    if (index_row == 0){
        double entropy=0;
        for (int i=0; i < MATRIX_ROW-2; i++){
            for (int j=i+1; j < MATRIX_ROW-1;j++){
                if (similarity_matrix[i*MATRIX_ROW + j] != 0.0)
                entropy = entropy + ((similarity_matrix[i*MATRIX_ROW + j]*log(similarity_matrix[i*MATRIX_ROW + j])) +  ((1-similarity_matrix[i*MATRIX_ROW + j]) * log(1-similarity_matrix[i*MATRIX_ROW + j])));
            }
        }
        entropy = (-1) * entropy;
        *final_entropy = entropy;

    }
}


__global__ void addMatrix (int *matrix,int *output)
{
    int sum=0;
    int index_row = (blockDim.x * blockIdx.x) + threadIdx.x;
    for (int column = 0; column < MATRIX_COLUNM; column++)
    {  
        for (int length = 0; length < STRING_LENGTH; length++)
        {
            sum = sum + matrix[index_row*MATRIX_COLUNM*STRING_LENGTH + column*STRING_LENGTH + length ];
            //sum = sum + *(*(*(matrix+index_row)+column)+length);
        }
    }

    output[index_row] = sum;
    sum = 0;
}


int main ()
{

    printf("****************************************************************************\n");
    printf("    BEGIN - Entropy Calculation for Dimensionality Reduction\n");
    printf("***************************************************************************\n\n");

    // Varibles for mesuered time elapsed
    struct timeval tval_before, tval_after, tval_result;
    gettimeofday(&tval_before, NULL);

    //reading CSV file
    readCSVFile(main_matrix);

    //Array that has all entropy values
    double vaulesOfEntropyWithoutColums [MATRIX_COLUNM+1];

    entropyCalculation(main_matrix, similarity_matrix,0,vaulesOfEntropyWithoutColums);

    printf("The entropy values for each column are as follows:\n");
    printf("    The value without the column %d: %f\n",0,vaulesOfEntropyWithoutColums[0]);

    //*******************************************************
    // BEGIN CUDA CODE
    //*******************************************************

    
    // Flating main matrix AND similarity matrix
    // varibles in host
    char *h_main_matrix = (char *) main_matrix;
    double *h_similarity_matrix = (double *) similarity_matrix;
    
    double h_entropy=0.0;
    // varibles in device
    char *d_main_matrix;
    double *d_similarity_matrix;
    double *d_entropy;

    //getting size of each varible
    int size_main_matrix = sizeof(main_matrix);
    int size_similarity_matrix = sizeof(similarity_matrix);
    int size_entropy = sizeof(double);

 
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
    err = cudaMemcpy(d_similarity_matrix, h_similarity_matrix, size_similarity_matrix, cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to copy vector h_similarity_matrix from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //For h_entropy
    err = cudaMemcpy(d_entropy, &h_entropy, size_entropy, cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to copy vector h_entropy from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //---------------------------
    // END  - COPY VAULE TO DEVICE
    //---------------------------


    //---------------------------
    // BEGIN - LAUNCH KERNEL
    //---------------------------

    // __global__ void entropyCalculation_CUDA (char *data_matrix, double *similarity_matrix, double *final_entropy,int withoutColum)
    entropyCalculation_CUDA<<<100,10>>>(d_main_matrix, d_similarity_matrix,d_entropy, 0);
    //---------------------------
    // END -  LAUNCH KERNEL
    //---------------------------


    //---------------------------
    // BEGIN - COPY VAULES TO HOST
    //---------------------------

    //For h_entropy
    err = cudaMemcpy(&h_entropy, d_entropy, size_entropy, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to copy vector d_entropy from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    //---------------------------
    // END -  COPY VAULES TO HOST
    //---------------------------
    


    //---------------------------
    // BEGIN - PRINT DATA FROM GPU
    //---------------------------

    //For h_entropy
    printf("\n");
    printf("The entropy values for each column are as follows:\n");
    printf("    The value with all columns    : %f\n",h_entropy);

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
    printf("    END - Entropy Calculation for Dimensionality Reduction\n");
    printf("***************************************************************************\n");
    printf("\nTime elapsed: %ld.%06ld\n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
    return 0;
}

