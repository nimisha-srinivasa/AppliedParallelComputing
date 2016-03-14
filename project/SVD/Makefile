CC      = nvcc
CFLAGS  = 
LDFLAGS  = -lcublas -lcusolver

#EXEC = hello_cuda.x dotprod.x add_vec_gpu_thd-blk.x
#EXEC = vec_add_cuda.x svd_cuda.x
EXEC = svd.x

all:  $(EXEC)

svd.x: my_svd.cu
	$(CC) -o $@ $^ $(LDFLAGS)


submit_svd:
	sbatch -v submit_svd.job

submit_test_qr:
	sbatch -v test_qr_example.job


clean:
	rm  $(EXEC)
