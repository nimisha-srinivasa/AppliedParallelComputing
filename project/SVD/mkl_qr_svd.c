#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include "mkl.h"

#define M 100000
#define N 300
#define FILENAME "data.txt"
#define LDA M
#define LDU M
#define LDVT N
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

/* DGESVD prototype 
extern void dgesvd( char* jobu, char* jobvt, int* m, int* n, double* a,
                int* lda, double* s, double* u, int* ldu, double* vt, int* ldvt,
                double* work, int* lwork, int* info ); */
/* Auxiliary routines prototypes */
void print_matrix( char* desc, int m, int n, double* a, int lda );

void fill(double *p, int n) {
    
    for (int i = 0; i < n; i++) {
        p[i] = (double) (2.0*drand48() + 1.0);
    }
}

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

void computeRfromA(double *A, double *R, int lda, int ldr){
    for(int i=0; i< ldr; i++){
        for(int j=0; j< ldr; j++){
            if( i <= j)
                R[i+j*ldr] = A[i+j*lda];
            else
                R[i+j*ldr] = 0.0f;
        }
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
        int m = M, n = N, lda = LDA, ldu = LDU, ldvt = LDVT, info, info_QR, lwork;
        int m_SVD=N;
        int n_SVD=N;
        int lda_SVD=n;
        int ldu_SVD=n;
        int ldvt_SVD=n;
        double wkopt;
        double* work;
        /* Local arrays */
        /*double s[N], u[LDU*M], vt[LDVT*N];*/
        double s[N], u[N*n], vt[N*N];
        double *a =(double *)malloc(LDA*N*sizeof(double));
        double *R =(double *)malloc(N*N*sizeof(double));
        double *tau =(double *)malloc(MIN(m,n)*sizeof(double));
        fill(a, m*n);
        /*readMatrixFromFile(a, lda);*/
        /* Executable statements */
        printf( " DGESVD Example Program Results\n" );
        /* Query and allocate the optimal workspace */
        /* QR */
        int lwork_QR=-1;
        double iwork_QR;
        dgeqrf_(&m, &n, a, &lda, tau, &iwork_QR, &lwork_QR, &info_QR);
        lwork_QR = (int)iwork_QR;
        double* work_QR = new double[lwork_QR];

        /* SVD*/
        lwork = -1;
        dgesvd( "All", "All", &m_SVD, &n_SVD, R, &lda_SVD, s, u, &ldu_SVD, vt, &ldvt_SVD, &wkopt, &lwork,
         &info );
        lwork = (int)wkopt;
        work = (double*)malloc( lwork*sizeof(double) );

        /*  measure time */
        double Mflop_s, seconds_QR, seconds_SVD;
        int n_iterations;
        /*seconds = read_timer( ); */
        /* computer QR first */
            seconds_QR = read_timer( );
            dgeqrf_(&m, &n, a, &lda, tau, work_QR, &lwork_QR, &info_QR);
            seconds_QR = read_timer( ) - seconds_QR;
            
        computeRfromA(a,R, m, n);
        for( n_iterations = 1; n_iterations<100 ; n_iterations ++ ) 
        {   
            
            
            seconds_SVD = read_timer( );
            dgesvd( "All", "All", &m_SVD, &n_SVD, R, &lda_SVD, s, u, &ldu_SVD, vt, &ldvt_SVD, work, &lwork,
             &info );
            seconds_SVD = read_timer( ) - seconds_SVD;
            
           

        }
        /*seconds = read_timer( ) - seconds;*/
        /*  compute Mflop/s rate */
        /*Mflop_s = 4e-6 * n_iterations * m * m * n / seconds;
        printf ("Mflop/s: %g\n", Mflop_s);*/
        

        /* Check for convergence */
        if( info > 0 ) {
                printf( "The algorithm computing SVD failed to converge.\n" );
                exit( 1 );
        }

        /* Print singular values */
        /*
        print_matrix( "Singular values", 1, n, s, 1 );
        print_matrix( "Left singular vectors (stored columnwise)", m, n, u, ldu );
        print_matrix( "Right singular vectors (stored rowwise)", n, n, vt, ldvt );
        */
        /*printf("total time taken is %lf\n",seconds);*/
        /*printf("QR: %lf\n", seconds_QR);*/
        printf("SVD: %lf\n", seconds_SVD);

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