#include <stdio.h>
#include <stdlib.h> 
#include <string.h>

int getNumberOfLines(FILE *fptr){
    int ans=0;
    ssize_t read;
    char *line=NULL;
    size_t len=0;
    int i=0;
    while ((read = getline(&line, &len, fptr)) != -1) {
        i++;
    }
    return i;
}

int getNumberOfColumns(FILE *fptr){
    int ans=0;
    size_t len=0;
    char *line=NULL;
    char *word=NULL;
    ssize_t read;
    if((fptr!=NULL) && ((read=getline(&line, &len, fptr)) != -1)){
        do{
            word=strsep(&line,",");
            ans++;
        }while(line!=NULL && word!=NULL);
    }
    return ans;
}

int freeData(float** matrix, int row, int col){
    int i=0;
    for(i = 0; i < row; i++)
        free(matrix[i]);
}

void printMatrix(float** matrix, int row, int col){
    int i,j;
    for(i=0;i<row;i++){
        for(j=0;j<col;j++){
            printf("%f\t",matrix[i][j]);
        }
        printf("\n");
    }
}

int main()
{

    FILE *myFile1;
    FILE *myFile2;
    FILE *myFile3;
    char *filename="c.txt";
    myFile1 = fopen(filename, "r");
    if (myFile1 == NULL)
    {
        printf("Error Reading File\n");
        exit (0);
    }
    printf("all is well \n");

    /*get number of lines in the file*/
    int r=getNumberOfLines(myFile1);
    fclose(myFile1);

    myFile2 = fopen(filename, "r");
    if (myFile2 == NULL)
    {
        printf("Error Reading File\n");
        exit (0);
    }
    int c=getNumberOfColumns(myFile2);
    fclose(myFile2);
    printf("r=%d and c=%d\n",r,c);
    char *line=NULL;
    char *word=NULL;
    int attr;
    int i,j;
    size_t len = 0;
    ssize_t read;
    printf("r=%d and c=%d\n",r,c);

    myFile3 = fopen(filename, "r");
    if (myFile3 == NULL)
    {
        printf("Error Reading File\n");
        exit (0);
    }
    float *dataset[r];
    for (i=0; i<r; i++)
         dataset[i] = (float *)malloc(c * sizeof(float));
    i=0;
    while ((read = getline(&line, &len, myFile3)) != -1) {
        
        j=0;

        do{
            word=strsep(&line,",");
            attr = atoi(word);
            dataset[i][j]=(float)attr;
            j++;
        }while(line!=NULL && word!=NULL);
        i++;           
        
    }
    fclose(myFile3);

    printMatrix(dataset, r,c);

    freeData(dataset, r,c);

    return 0;
}


