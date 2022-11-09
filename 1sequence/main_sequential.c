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

#define MATRIX_ROW 1000
#define MATRIX_COLUNM 6
#define STRING_LENGTH 4
#define MAX_LEN_LINE STRING_LENGTH*MATRIX_COLUNM

#define PRINT_LOGS 0
#define PRINT_LOGS_MATRIX 0

//Matrix type double (*)[MATRIX_ROW]
double similarity_matrix[MATRIX_ROW][MATRIX_ROW]={0.0};

//Matrix type char (*)[MATRIX_COLUNM][STRING_LENGTH]
char main_matrix[MATRIX_ROW][MATRIX_COLUNM][STRING_LENGTH]={/*  */};


// function to calculate Hamming distance
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

void printDataMatrix(char data_matrix[][MATRIX_COLUNM][STRING_LENGTH]){
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

void readCSVFile (char main_matrix[MATRIX_ROW][MATRIX_COLUNM][STRING_LENGTH]){
    
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

void printSimilarityMatrix( double similarity_matrix[MATRIX_ROW][MATRIX_ROW]){

    // Print similarity matrix
    printf("Printing similarity matrix:\n");
    for(int i=0; i<MATRIX_ROW; i++){
        for (int j=0; j<MATRIX_ROW; j++){
            printf("%f ", similarity_matrix[i][j]);
        }
        printf("\n");
    }

}


double entropyCalculation (char data_matrix[][MATRIX_COLUNM][STRING_LENGTH],double similarity_matrix[][MATRIX_ROW],int withoutColum){
    
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
    return entropy;
}

void entropyCalculation2 (char data_matrix[][MATRIX_COLUNM][STRING_LENGTH],double similarity_matrix[][MATRIX_ROW],int withoutColum, double *final_entropy){
    
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

int main (){

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

    for (int column = 0; column < MATRIX_COLUNM+1; column++){
        //entropyCalculation(main_matrix,0) - 0 means with all columns
        //vaulesOfEntropyWithoutColums[column] = entropyCalculation(main_matrix, similarity_matrix, column);
        entropyCalculation2(main_matrix, similarity_matrix, column,vaulesOfEntropyWithoutColums);
    }

    printf("\n");
    printf("The entropy values for each column are as follows:\n");
    for (int i = 0; i < MATRIX_COLUNM+1; i++){
        if (i==0) // 0 in is saved the entropy vaule with all the columns
            printf("    The value with all columns    : %f\n",vaulesOfEntropyWithoutColums[i]);

        else
            printf("    The value without the column %d: %f\n",i,vaulesOfEntropyWithoutColums[i]);
    }

    // getting Time elapsed
    gettimeofday(&tval_after, NULL);
    timersub(&tval_after, &tval_before, &tval_result);
    
    printf("\n****************************************************************************\n");
    printf("    END - Entropy Calculation for Dimensionality Reduction\n");
    printf("***************************************************************************\n");
    printf("\nTime elapsed: %ld.%06ld\n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
    return 0;
}

