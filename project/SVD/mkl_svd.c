#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include "mkl.h"

#define M 31568
#define N 51
#define FILENAME "data.txt"
#define LDA M
#define LDU M
#define LDVT N

/* DGESVD prototype 
extern void dgesvd( char* jobu, char* jobvt, int* m, int* n, double* a,
                int* lda, double* s, double* u, int* ldu, double* vt, int* ldvt,
                double* work, int* lwork, int* info ); */
/* Auxiliary routines prototypes */
void print_matrix( char* desc, int m, int n, double* a, int lda );

void readMatrixFromFile(double *p, int lda){
    FILE *myFile;
    char *filename=FILENAME;
    myFile = fopen(filename, "r");
    if (myFile == NULL)
    {
        printf("Error Reading File\n");
        exit (0);
    }

    char *line=NULL;
    char *word=NULL;
    double attr;
    size_t len = 0;
    ssize_t read;
    int row,col;

    /*fill the matrix*/
    row=0;
    while (((read = getline(&line, &len, myFile)) != -1) && row<M) {
        col=0;
        do{
            word=strsep(&line,",");
            attr = atof(word);
            p[row + col*lda]=attr;
            col++;
        }while(line!=NULL && word!=NULL && col<N);
        row++;        
    }  
}

double read_timer( )
{
    static bool initialized = false;
    static struct timeval start;
    struct timeval end;
    if( !initialized )
    {
        gettimeofday( &start, NULL );
        initialized = true;
    }

    gettimeofday( &end, NULL );

    return (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
}

int main() {
        /* Locals */
        int m = M, n = N, lda = LDA, ldu = LDU, ldvt = LDVT, info, lwork;
        double wkopt;
        double* work;
        /* Local arrays */
        double s[N], u[LDU*M], vt[LDVT*N];
        double *a =(double *)malloc(LDA*N*sizeof(double));
        readMatrixFromFile(a, lda);
        /* Executable statements */
        printf( " DGESVD Example Program Results\n" );
        /* Query and allocate the optimal workspace */
        lwork = -1;
        dgesvd( "All", "All", &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, &wkopt, &lwork,
         &info );
        lwork = (int)wkopt;
        work = (double*)malloc( lwork*sizeof(double) );

        /*  measure time */
        double Mflop_s, seconds;
        int n_iterations;
        seconds = read_timer( );
        for( n_iterations = 1; n_iterations<10 ; n_iterations ++ ) 
        {
            /* Compute SVD */
            dgesvd( "All", "All", &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork,
             &info );

        }
        seconds = read_timer( ) - seconds;
        /*  compute Mflop/s rate */
        Mflop_s = 4e-6 * n_iterations * m * m * n / seconds;
        printf ("Mflop/s: %g\n", Mflop_s);
        

        /* Check for convergence */
        if( info > 0 ) {
                printf( "The algorithm computing SVD failed to converge.\n" );
                exit( 1 );
        }
        /* Print singular values */
        print_matrix( "Singular values", 1, n, s, 1 );
        /* Print left singular vectors */
        print_matrix( "Left singular vectors (stored columnwise)", m, n, u, ldu );
        /* Print right singular vectors */
        print_matrix( "Right singular vectors (stored rowwise)", n, n, vt, ldvt );
        printf("time take is %lf\n",seconds);
        /* Free workspace */
        free( (void*)work );
        exit( 0 );
} /* End of DGESVD Example */

/* Auxiliary routine: printing a matrix */
void print_matrix( char* desc, int m, int n, double* a, int lda ) {
        int i, j;
        printf( "\n %s\n", desc );
        for( i = 0; i < m; i++ ) {
                for( j = 0; j < n; j++ ) printf( " %6.2f", a[i+j*lda] );
                printf( "\n" );
        }
}