CUDA_CC      = nvcc
CFLAGS  = 
CUDA_LDFLAGS  = -lcublas -lcusolver

#Comet
#MKLPATH=/opt/intel/Compiler/11.1/072/mkl
MKLPATH=$MKL_ROOT
MKL_CC=icpc
CFLAGS = -I$(MKLPATH)/include  

MKLFLAGS=  -I$MKL_ROOT/include ${MKL_ROOT}/lib/intel64/libmkl_scalapack_lp64.a -Wl,--start-group ${MKL_ROOT}/lib/intel64/libmkl_intel_lp64.a ${MKL_ROOT}/lib/intel64/libmkl_core.a ${MKL_ROOT}/lib/intel64/libmkl_sequential.a -Wl,--end-group ${MKL_ROOT}/lib/intel64/libmkl_blacs_intelmpi_lp64.a 

MKL_LDFLAGS = $(MKLFLAGS) -lpthread -lm 

#EXEC = hello_cuda.x dotprod.x add_vec_gpu_thd-blk.x
#EXEC = vec_add_cuda.x svd_cuda.x
EXEC = svd.x mkl_svd

all:  $(EXEC)

svd.x: gpu_svd.cu
	$(CUDA_CC) -o $@ $^ $(CUDA_LDFLAGS)

mkl_svd: mkl_svd.o
	$(MKL_CC) -o $@ $^ $(MKL_LDFLAGS)


submit_gpu:
	sbatch -v submit_svd.job

submit_mkl:
	sbatch -v submit_mkl.sh

%.o: %.c
	$(MKL_CC) $(CFLAGS) -c $<


clean:
	rm  $(EXEC) *.o
