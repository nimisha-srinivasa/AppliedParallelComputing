CUDA_CC      = nvcc
CFLAGS  = 
CUDA_LDFLAGS  = -lcublas -lcusolver


#EXEC = hello_cuda.x dotprod.x add_vec_gpu_thd-blk.x
#EXEC = vec_add_cuda.x svd_cuda.x
EXEC = svd.x

all:  $(EXEC)

svd.x: gpu_svd.cu
	$(CUDA_CC) -o $@ $^ $(CUDA_LDFLAGS)




submit_gpu:
	qsub submit.pbs



clean:
	rm  $(EXEC) *.o
