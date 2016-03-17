/* Compute the SVD of a matrix */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#define ROWS 100000
#define COLS 300
//#define ROWS 31568
//#define COLS 51

#define FILENAME "data.txt"
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define TOTAL_ITERATIONS 5

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

void fill(float *p, int n) {
    // This will be replaced by retrieving the data...
    for (int i = 0; i < n; i++) {
        p[i] = (float) (2.0*drand48() + 1.0);
    }
}

void print_matrix(int m, int n, float *A, int lda, const char *name) {
    printf("================A===============================\n");
    for(int row = 0; row < m; row++) {
        for(int col = 0; col < n; col++) {
            float Areg = A[row + col*lda];
            printf("%f\t", Areg);
        }
        printf("\n");
    }
    printf("================end of A===============================\n");
}

void readMatrixFromFile(float *p, int lda){
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
    float attr;
    size_t len = 0;
    ssize_t read;
    int row,col;

    //fill the matrix
    row=0;
    while (((read = getline(&line, &len, myFile)) != -1) && row<ROWS) {
        col=0;
        do{
            word=strsep(&line,",");
            attr = atof(word);
            p[row + col*lda]=attr;
            col++;
        }while(line!=NULL && word!=NULL && col<COLS);
        row++;        
    }  
}

void computeRfromA(float *A, float *R, int lda, int ldr){
    for(int i=0; i< ldr; i++){
        for(int j=0; j< ldr; j++){
            if( i <= j)
                R[i+j*ldr] = A[i+j*lda];
            else
                R[i+j*ldr] = 0.0f;
        }
    }
}

int main(int argc, char *argv[])
{
    
    printf("with my modifications \n");
    cusolverDnHandle_t cudenseH = NULL;

    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess; 

    /*====================== used for timing purposes   ======================  */

    cudaEvent_t start, stop;
    float time_SVD=0.0f;
    float time_temp;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    /* ======================   Matrix definitions ====================== */

    const int rows = ROWS;
    const int cols = COLS;
    const int mat_A_size = rows*cols;
    const int mat_Q_size = rows*cols;
    const int mat_R_size = cols*cols;
    const int mat_TAU_size = MIN(rows,cols);
    const int mat_U_size = cols*cols;
    const int mat_S_size = cols;
    const int mat_VT_size = cols*cols;

    size_t size_A = mat_A_size*sizeof(float);
    size_t size_TAU = mat_TAU_size*sizeof(float);
    size_t size_R = mat_R_size*sizeof(float);
    size_t size_S = mat_S_size*sizeof(float);
    size_t size_U = mat_U_size*sizeof(float);
    size_t size_VT = mat_VT_size*sizeof(float);

    float *h_A = (float*)malloc(size_A);
    float *h_R = (float*)malloc(size_R);

    /*  copy back only S and Vt after n_iterations  */
    float *h_S = (float*)malloc(size_S);
    float *h_U = (float*)malloc(size_U);
    float *h_VT = (float*)malloc(size_VT);

    /* ====================== matrixes in device ====================== */
    float *d_A = NULL;
    float *d_R = NULL;
    float *d_TAU_QR = NULL;
    float *d_work_QR = NULL;
    float *d_U = NULL;
    float *d_S = NULL;
    float *d_VT = NULL;
    float *d_work_SVD = NULL;
    float *r_work_SVD = NULL;

    /* other variables required for computations */
    int *devInfo_QR = NULL; 
    int *devInfo_SVD = NULL; 
    int info_gpu_QR = 0;
    int info_gpu_SVD =0;
    int lwork_size_QR = 0;
    int lda_QR = rows;
    int lda_SVD=cols;
    int lwork_size_SVD = 0;

    
    

    //fill(h_A, mat_A_size);
    readMatrixFromFile(h_A, rows);

    /*
    printf("A\n");
    print_matrix(rows, cols, h_A, rows, "A");
    printf("\n\n\n");
    */

    /*  ====================== initialise CUDA handle =========================== */

    cusolver_status = cusolverDnCreate(&cudenseH);
    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);


    /* ====================Create data structures for device ==================== */

    /* for QR */
    cudaStat1 = cudaMalloc((void**)&d_A, size_A);
    assert(cudaSuccess == cudaStat1);

    cudaStat1 = cudaMalloc((void**)&d_R, size_R);
    assert(cudaSuccess == cudaStat1);

    cudaStat1 = cudaMalloc ((void**)&d_TAU_QR, size_TAU);   
    assert(cudaSuccess == cudaStat1); 

    cudaStat1 = cudaMalloc((void**)&devInfo_QR, sizeof(int));
    assert(cudaSuccess == cudaStat1);


    /* for SVD */
    cudaStat1 = cudaMalloc((void**)&d_U, size_U);
    assert(cudaSuccess == cudaStat1);

    cudaStat1 = cudaMalloc((void**)&d_S, size_S);
    assert(cudaSuccess == cudaStat1);

    cudaStat1 = cudaMalloc((void**)&d_VT, size_VT);
    assert(cudaSuccess == cudaStat1);

    
    cudaStat1 = cudaMalloc((void**)&devInfo_SVD, sizeof(int));
    assert(cudaSuccess == cudaStat1);


    /* ======================copy data to device ======================*/

    cudaStat1 = cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat1);


    /*  ====================== Compute QR ======================    */


    /* calculate the sizes needed for pre-allocated buffer Lwork  */
    cusolver_status = cusolverDnSgeqrf_bufferSize(cudenseH, rows, cols, d_A, lda_QR, &lwork_size_QR );
    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);

    cudaStat1 = cudaMalloc((void**)&d_work_QR, lwork_size_QR);
    assert(cudaSuccess == cudaStat1);

    cusolverDnSgeqrf( cudenseH, rows, cols, d_A, lda_QR, d_TAU_QR, d_work_QR, lwork_size_QR, devInfo_QR );

    /* check if QR is good or not  */
    cudaStat1 = cudaMemcpy(&info_gpu_QR, devInfo_QR, sizeof(int), cudaMemcpyDeviceToHost); 
    assert(cudaSuccess == cudaStat1);
    assert(0 == info_gpu_QR);

    /* copy A to host */
    cudaStat1 = cudaMemcpy(h_A, d_A, size_A, cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);

    computeRfromA(h_A, h_R, lda_QR, lda_SVD);

    /*  ====================== End of QR ======================    */


    /*  ====================== Compute SVD ======================    */

    int rows_SVD=cols;
    int cols_SVD=cols;
    int ldu_SVD=cols;
    int ldvt_SVD=cols;
    char jobu = 'A'; // We do not want/need U
    char jobvt = 'A'; // We want all the vectors of VT

    /*compute buffer size for SVD */
    cusolver_status = cusolverDnSgesvd_bufferSize(cudenseH, rows_SVD, cols_SVD, &lwork_size_SVD );
    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);

    /* for timing purposes only */
    int total_iterations = TOTAL_ITERATIONS;
    cudaEvent_t events_start[total_iterations];
    cudaEvent_t events_stop[total_iterations];
    for(int i=0;i<total_iterations;i++){
        cudaEventCreate(&events_start[i]);
        cudaEventCreate(&events_stop[i]);
    }
    int n_iterations;
    for(n_iterations =0; n_iterations < total_iterations; n_iterations ++){

        cudaStat1 = cudaMemcpy(d_R, h_R, size_R, cudaMemcpyHostToDevice);
        assert(cudaSuccess == cudaStat1); 

        cudaStat1 = cudaMalloc((void**)&d_work_SVD, lwork_size_SVD);
        assert(cudaSuccess == cudaStat1);  

        cudaStat1 = cudaMalloc((void**)&r_work_SVD, lwork_size_SVD);
        assert(cudaSuccess == cudaStat1);       

        cudaEventRecord(events_start[n_iterations], 0);
        cusolver_status = cusolverDnSgesvd (cudenseH, jobu, jobvt, rows_SVD, cols_SVD, d_R, lda_SVD, d_S, d_U, ldu_SVD, d_VT, ldvt_SVD, d_work_SVD, lwork_size_SVD, r_work_SVD, devInfo_SVD);

        cudaEventRecord(events_stop[n_iterations], 0);
        cudaEventSynchronize(events_stop[n_iterations]);

        /* check if SVD is good or not  */
        cudaStat1 = cudaMemcpy(&info_gpu_SVD, devInfo_SVD, sizeof(int), cudaMemcpyDeviceToHost); 
        assert(cudaSuccess == cudaStat1);
        assert(0 == info_gpu_SVD);

        /* calculate time for this iteration */
        cudaEventElapsedTime(&time_temp, events_start[n_iterations], events_stop[n_iterations]);
        /*
        printf ("Time for the kernel: %f ms\n", time_temp);
        printf ("\n\n\n");
        */

        time_SVD+=time_temp;

        /* free cudaResources */
        if (d_work_SVD ) cudaFree(d_work_SVD);

    }

    float Mflop_rate;
    printf ("Time for the kernel: %f ms\n", time_SVD);
    Mflop_rate = 1e-6 * 4 * cols * cols * cols * n_iterations / time_SVD;
    printf ("n_iterations = %d\n",n_iterations);
    printf ("Mflop/s: %f\n", Mflop_rate);

    printf("cusolverDnSgesvd status :\t");
    switch(cusolver_status)
      {
        case CUSOLVER_STATUS_SUCCESS:
          printf("success\n");
          break;
        case CUSOLVER_STATUS_NOT_INITIALIZED :
          printf("Library cuSolver not initialized correctly\n");
          break;
        case CUSOLVER_STATUS_INVALID_VALUE:
          printf("Invalid parameters passed\n");
          break;
        case CUSOLVER_STATUS_INTERNAL_ERROR:
          printf("Internal operation failed\n");
          break;
        case CUSOLVER_STATUS_EXECUTION_FAILED:
          printf("Execution failed\n");
          break;
      }


    /* ================END of SVD Computation======================= */

    /*  check if SVD is good or not  */
    cudaStat1 =cudaMemcpy(&info_gpu_SVD,devInfo_SVD,sizeof(int),cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);
    printf("after SVD: info_gpu = %d\n", info_gpu_SVD);
    assert(0 == info_gpu_SVD); 
    
    ///*  copy the solutions back to the host */
    cudaStat1 = cudaMemcpy(h_R, d_R, size_R, cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);

    cudaStat1 = cudaMemcpy(h_U, d_U, size_U, cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1); 
    cudaStat1 = cudaMemcpy(h_S, d_S, size_S, cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);
    cudaStat1 = cudaMemcpy(h_VT, d_VT, size_VT, cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);

    /* 
    printf("U\n");
    print_matrix(rows, rows, h_U, rows, "A");
    printf("\n\n\n");

    printf("S\n");
    print_matrix(rows, cols, h_S, rows, "S");
    printf("\n\n\n");

    printf("VT\n");
    print_matrix(cols, cols, h_VT, cols, "VT");
    printf("\n\n\n");

     
    printf("A\n");
    print_matrix(rows, rows, h_A, rows, "A");
    printf("\n\n\n");
    */

    /* free resources */
    if (d_A ) cudaFree(d_A);
    if (d_TAU_QR ) cudaFree(d_TAU_QR);
    if (d_R ) cudaFree(d_R);
    if (d_S ) cudaFree(d_S);
    if (d_U ) cudaFree(d_U);
    if (d_VT ) cudaFree(d_VT);

    if(h_A) free(h_A);
    if(h_R) free(h_R);
    if(h_S) free(h_S);
    if(h_U) free(h_U);
    if(h_VT) free(h_VT);

    if (cudenseH) cusolverDnDestroy(cudenseH);

    

    cudaDeviceReset();
}
