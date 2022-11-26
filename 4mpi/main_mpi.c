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
* November 2022
*******************************************************************************/
//mpicc main_mpi.c -o main_mpi.o -lm

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <mpi.h>

#define MATRIX_ROW 2560
#define MATRIX_COLUMN 10
#define STRING_LENGTH 4
#define MAX_LEN_LINE STRING_LENGTH*MATRIX_COLUMN

#define PRINT_LOGS 0
#define PRINT_LOGS_MATRIX 0
#define  NUM_PROCS 5

//MPI
#define send_data_tag 2001
#define return_data_tag 2002

//Matrix type double (*)[MATRIX_ROW]
double similarity_matrix[MATRIX_ROW][MATRIX_ROW]={0.0};

//Matrix type char (*)[MATRIX_COLUMN][STRING_LENGTH]
char main_matrix[MATRIX_ROW][MATRIX_COLUMN][STRING_LENGTH]={/*  */};

// function to calculate Hamming distance
int hammingDist(char* str1, char* str2){
    int i = 0, count = 0;
    while (str1[i] != '\0') {
        if (str1[i] == str2[i] && str1[i] != '\n' && str2[i] != '\n')
            count++;
        i++;
    }
    return count;
}

void printDataMatrix(char data_matrix[][MATRIX_COLUMN][STRING_LENGTH]){
    // Print read data
    printf("Printing read matrix:\n");
    for(int i=0; i<MATRIX_ROW; i++){
        for (int j=0; j<MATRIX_COLUMN; j++){
            printf("%s ", data_matrix[i][j]);
        }
        //printf(" %d",i);
        printf("\n");
    }
}

void readCSVFile (char main_matrix[MATRIX_ROW][MATRIX_COLUMN][STRING_LENGTH]){
    
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
        for (int colunm = 0; colunm < MATRIX_COLUMN; colunm++)
        {
            if (colunm == MATRIX_COLUMN-1){
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

double entropyCalculation (char data_matrix[][MATRIX_COLUMN][STRING_LENGTH],double similarity_matrix[][MATRIX_ROW],int withoutColum){
    
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

            for (int column_d=0; column_d < MATRIX_COLUMN; column_d++){
                if (withoutColum == 0){
                   sum = sum + hammingDist(data_matrix[row_s][column_d],data_matrix[column_s][column_d]);
                }
                else{
                    if (withoutColum-1 != column_d)
                        sum = sum + hammingDist(data_matrix[row_s][column_d],data_matrix[column_s][column_d]);

                }
                
            }
            if (withoutColum == 0)
                similarity_matrix[row_s][column_s]= (double) sum/MATRIX_COLUMN;
            
            else
                similarity_matrix[row_s][column_s]= (double) sum/(MATRIX_COLUMN-1);
            

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

int main(int argc, char **argv){

    // Varibles for mesuered time elapsed
    struct timeval tval_before, tval_after, tval_result;
    gettimeofday(&tval_before, NULL);

    //reading CSV file
    readCSVFile(main_matrix);

    //*******************************************************
    // BEGIN MPI CODE
    //*******************************************************
    
    // -----------------------------
    // Varibles MPI for all nodes
    // -----------------------------
    int i, ierr,my_id, index;
    MPI_Status status;

    // Vector that contains ids for columns to operate
    int columns_to_operate[MATRIX_COLUMN]={0};
    int columns_operated[MATRIX_COLUMN];

    // Vector that contains vaules of entropy
    double values_entropy_calculated[MATRIX_COLUMN];


    // -----------------------------
    // Varibles MPI for master node
    // -----------------------------

    int root_process, id_proc, num_column_each, num_columns_remainder, num_procs;

    int columns_ids[MATRIX_COLUMN];

    //Array that has all entropy values
    double vaulesOfEntropyWithoutColums [MATRIX_COLUMN+1];

    // Begin parallel processes
    ierr = MPI_Init(&argc, &argv);

    // get ID of current process
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
    root_process = 0;

    //-----------------------------------
    // Following code is for ROOT node
    // ---------------------------------
    if (my_id == root_process){
        printf("----------------------\nI am the MASTER proc\n----------------------\n");
        printf("***************************************************************************\n");
        printf("    BEGIN - Entropy Calculation for Dimensionality Reduction with MPI\n");
        printf("***************************************************************************\n\n");

        // get numbers of processes
        ierr = MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

        // calculation number of columns for each node
        num_column_each = MATRIX_COLUMN / num_procs; // quotient 
        num_columns_remainder = MATRIX_COLUMN % num_procs; // remainder 

        // Filling columns_ids vector with id of colunms 
        for ( i = 0; i < MATRIX_COLUMN; i++){
            columns_ids[i]= i+1;
        }


        //-------------------------------------------
        // Distribute columns into slaves nodes
        //-------------------------------------------
        for (id_proc=1 ; id_proc < num_procs; id_proc++ ){

            // distribute quotient columns
            for (index = 0; index < num_column_each; index++ ){
                columns_to_operate[index] = columns_ids[id_proc*num_column_each+index];
            }

            // distribute remainder columns
            if (num_columns_remainder > 0 && id_proc <= num_columns_remainder){
                columns_to_operate[num_column_each] = columns_ids[num_procs*num_column_each+id_proc-1];
            }
            
            // Send messages to slaves nodes
            /* A vector with colunms to operate for eache colunm will sent.
             And always the vector sent will have size of number of colunm. 
             Each slave node will know which columns to operate on when it finds a 
             0 at the end.*/

             //http://condor.cc.ku.edu/~grobe/docs/intro-MPI-C.shtml


            // send columns_to_operate for id_proc node
            ierr = MPI_Send(columns_to_operate, MATRIX_COLUMN, MPI_INT, 
                        id_proc, send_data_tag, MPI_COMM_WORLD);

            // cleaning columns_to_operate for the other slave node
            for (i = 0; i < num_column_each+1; i++){
                columns_to_operate[i]= 0;
            }

        }

        //-------------------------------------------
        // Columns for master node and calculation
        //-------------------------------------------

        // Columns for master
        for (index = 0; index < num_column_each; index++ ){
            columns_to_operate[index] = columns_ids[index];
        }

        /* Running entropy algorithm for columns_to_operate and colunm 0 wich is 
        entropy for all matrix with all colunm in it. */

        // this is for entropy of all matrix. Master always will do this calculation
        vaulesOfEntropyWithoutColums[0]= entropyCalculation(main_matrix, similarity_matrix, 0);
        index = 0;
        while (columns_to_operate[index] != 0){
        vaulesOfEntropyWithoutColums[columns_to_operate[index]] = entropyCalculation(main_matrix, similarity_matrix, columns_to_operate[index]);
        index++; 
        }
        
        //------------------------------------------------
        // COLLET values of entropy from the slaves nodes
        //------------------------------------------------
        for (id_proc=1 ; id_proc < num_procs; id_proc++ ){

            // cleaning values of columns_operated and values_entropy_calculated
            for (i = 0; i < num_column_each+1; i++){
                columns_operated[i]= 0;
                values_entropy_calculated[i]= 0;
            }

            // recived IDs colunms operated by id_proc 
            ierr = MPI_Recv(columns_operated, MATRIX_COLUMN, MPI_INT, 
                id_proc, return_data_tag, MPI_COMM_WORLD, &status);


            // recived all values of entropy calculated for each node
            ierr = MPI_Recv(values_entropy_calculated, MATRIX_COLUMN, MPI_DOUBLE, 
                id_proc, return_data_tag, MPI_COMM_WORLD, &status);

            // Storage values of entropy calculated 
            index = 0;
            while (columns_operated[index] != 0){
                vaulesOfEntropyWithoutColums[columns_operated[index]] = values_entropy_calculated[index];
                index++;
            }

        }
        
        //---------------------------------
        // PRINT  values of entropy
        //---------------------------------

        printf("\n");
        printf("The entropy values for each column are as follows:\n");
        for (int i = 0; i < MATRIX_COLUMN+1; i++){
            if (i==0) // 0 in is saved the entropy vaule with all the columns
                printf("    The value with all columns    : %f\n",vaulesOfEntropyWithoutColums[i]);
            else
                printf("    The value without the column %d: %f\n",i,vaulesOfEntropyWithoutColums[i]);
        }

        // getting Time elapsed
        gettimeofday(&tval_after, NULL);
        timersub(&tval_after, &tval_before, &tval_result);
        printf("\n***************************************************************************\n");
        printf("    END - Entropy Calculation for Dimensionality Reduction with MPI\n");
        printf("***************************************************************************\n");
        printf("\nTime elapsed: %ld.%06ld\n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
    }


    //--------------------------------------
    // Following code is for SLAVES nodes
    //-------------------------------------
    else{
        
        // Intialization and cleaning coloumns to operate
        for (i = 0; i < num_column_each+1; i++){
            columns_to_operate[i]= 0;
            values_entropy_calculated[i]=0;
        }
        
        // Receive columns to operate
        ierr =  MPI_Recv(columns_to_operate, MATRIX_COLUMN, MPI_INT, 
        root_process, send_data_tag, MPI_COMM_WORLD, &status); 

        // Running entropy function for columns to operate
        index = 0;
        while (columns_to_operate[index] != 0){
        values_entropy_calculated[index] = entropyCalculation(main_matrix, similarity_matrix, columns_to_operate[index]);
        index++; 
        }

        // Send columns to operate
        ierr = MPI_Send(columns_to_operate, MATRIX_COLUMN, MPI_INT, 
        root_process, return_data_tag, MPI_COMM_WORLD);


        // Send values of entropy
        ierr = MPI_Send(values_entropy_calculated, MATRIX_COLUMN, MPI_DOUBLE, 
        root_process, return_data_tag, MPI_COMM_WORLD);

    }
    


    //*******************************************************
    // END MPI CODE
    //*******************************************************
    ierr = MPI_Finalize();


}