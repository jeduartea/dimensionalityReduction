
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


#define MAX_LINES 10
#define VARIBLES_NUM 70
#define MAX_LEN MAX_LINES - 1

double similarityMatrix[MAX_LINES][MAX_LINES];
char dataMatrix[MAX_LINES][MAX_LEN];
char dataMatrix2[MAX_LINES][VARIBLES_NUM];

// function to calculate Hamming distance
int hammingDist(char* str1, char* str2)
{
    int i = 0, count = 0;
    while (str1[i] != '\0') {
        if (str1[i] == str2[i] && str1[i] != ',' && str1[i] != '\n')
            count++;
        i++;
    }
    return count;
}


int main(void)
{
    ///////////////////////////////////////////
    // BEGIN - read file name matrix.txt
    //////////////////////////////////////////
    printf("-----------------------------------------\n");
    printf("BEGIN - read file name matrix.txt\n");
    printf("-----------------------------------------\n");

    FILE *file;
    file = fopen("matrix2.txt", "r");


    if (file == NULL)
    {
    printf("Error opening file.\n");
    return 1;
    }

    // line will keep track of the number of lines read so far from the file
    int line = 0;


    while (!feof(file) && !ferror(file)){
    if (fgets(dataMatrix[line], MAX_LEN, file) != NULL) line++;
    }
    // Close the file when we are done working with it.
    fclose(file);


    // Print file read
    printf("Print content of file read:\n");
    for (int i = 0; i < line; i++){
    printf("%s", dataMatrix[i]);
    }
    printf("\n");

    printf("-----------------------------------------\n");
    printf("END - read file name matrix.txt\n");
    printf("-----------------------------------------\n");
    ///////////////////////////////////////////
    // END - read file name matrix.txt
    //////////////////////////////////////////


    ///////////////////////////////////////////
    // BEGIN - Entropy algorithm
    //////////////////////////////////////////

    printf("-----------------------------------------\n");
    printf("BEGIN - Entropy algorithm\n");
    printf("-----------------------------------------\n");

    // Calculation of the similarity matrix
    printf("Begin calculation of the similarity matrix...\n\n");
    double sum=0;
    for(int i=0; i<MAX_LINES; i++){
        for(int j=0; j< MAX_LINES; j++){
            similarityMatrix[i][j]= (double) hammingDist(dataMatrix[i],dataMatrix[j])/VARIBLES_NUM;
            sum=0;
        }
    }

    // Print similarity matrix
    printf("Printing similarity matrix:\n");
    for(int i=0; i<MAX_LINES; i++){
        for (int j=0; j<MAX_LINES; j++){
            printf("%f ", similarityMatrix[i][j]);
        }
        printf("\n");
    }

    // Print similarity matrix
    printf("Begin calculation of entroy...\n");
    double entropy=0;
    for (int i=0; i < MAX_LINES-2; i++){
        for (int j=i+1; j < MAX_LINES-1;j++){
            if (similarityMatrix[i][j] != 0.0)
               entropy = entropy + ((similarityMatrix[i][j]*log(similarityMatrix[i][j])) +  ((1-similarityMatrix[i][j]) * log(1-similarityMatrix[i][j])));
        }
    }
    entropy = (-1)* entropy;
    printf("The entropy value is: %f\n",entropy);

    printf("-----------------------------------------\n");
    printf("END - Entropy algorithm\n");
    printf("-----------------------------------------\n");
    ///////////////////////////////////////////
    // BEGIN READ FILE - Entropy algorithm
    //////////////////////////////////////////

    
  return 0;
}