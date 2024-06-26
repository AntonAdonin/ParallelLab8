#include <iostream>
#include <boost/program_options.hpp>
#include <omp.h>
#include <new>
#include <nvtx3/nvToolsExt.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <fstream>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_store.cuh>

#define OFFSET(x, y, m) (((x) * (m)) + (y))


namespace po = boost::program_options;

// cuda unique_ptr
template<typename T>
using cuda_unique_ptr = std::unique_ptr<T,std::function<void(T*)>>;

// new
template<typename T>
T* cuda_new(size_t size)
{
    T *d_ptr;
    cudaMalloc((void **)&d_ptr, sizeof(T) * size);
    return d_ptr;
}

// delete
template<typename T>
void cuda_delete(T *dev_ptr)
{
    cudaFree(dev_ptr);
}


__global__ void subtractArrays(const double *A, const double *Anew, double *Sub_res , int m) {

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  if ((i > 1) && (i < m) && (j > 1) && (j < m)) {
	    Sub_res[OFFSET(i,j,m)] = fabs(A[OFFSET(i,j,m)] - Anew[OFFSET(i,j,m)]);
  }
}


void printCudaError(cudaError_t error, char err_src[]) { //error printing function to reduce line count
    if (error != cudaSuccess) {
        printf("Error: %i while performing %s \n", error, err_src);
        exit(EXIT_FAILURE);
    }
}




int main(int argc, char **argv)
{
    int m = 256;
    int iter_max = 1000000;
    double tol = 1.0e-6;
    double error = 1.0;
    po::options_description desc("Options");
    desc.add_options()
        ("help", "print help")
        ("error,e", po::value<double>(&tol)->default_value(tol), "min error")
        ("size,n", po::value<int>(&m)->default_value(m), "size of grid")
        ("iterations,i", po::value<int>(&iter_max)->default_value(iter_max), "number of iterations");

    // Парсинг аргументов командной строки
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    int n = m;

    std::unique_ptr<double[]> A_ptr(new double[m*m]);
    std::unique_ptr<double[]> Anew_ptr(new double[m*m]);
    std::unique_ptr<double[]> Subtract_temp_ptr(new double[m*m]);
    double *temp;

    double* A = A_ptr.get();
    double* Anew = Anew_ptr.get();
    double* Subtract_temp = Subtract_temp_ptr.get();

    nvtxRangePushA("init");
    double corners[] = {10.0, 20.0, 30.0, 20.0};
    memset(A, 0, n * n * sizeof(double));
    A[0] = corners[0];
    A[n - 1] = corners[1];
    A[n * n - 1] = corners[2];
    A[n * (n - 1)] = corners[3];
    double number_of_steps = n - 1;
    double top_side_step = (double)abs(corners[0] - corners[1]) / number_of_steps;
    double right_side_step = (double)abs(corners[1] - corners[2]) / number_of_steps;
    double bottom_side_step = (double)abs(corners[2] - corners[3]) / number_of_steps;
    double left_side_step = (double)abs(corners[3] - corners[0]) / number_of_steps;

    double top_side_min = std::min(corners[0], corners[1]);
    double right_side_min = std::min(corners[1], corners[2]);
    double bottom_side_min = std::min(corners[2], corners[3]);
    double left_side_min = std::min(corners[3], corners[0]);
    for (int i = 1; i < n - 1; i ++) {
        A[i] = top_side_min + i * top_side_step;
        A[n * i] = left_side_min + i * left_side_step;
        A[(n-1) + n * i] = right_side_min + i * right_side_step;
        A[n * (n-1) + i] = bottom_side_min + i * bottom_side_step;
    }
    std::memcpy(Anew, A, n * n * sizeof(double));
    nvtxRangePop();


    dim3 grid(32 , 32);

	dim3 block(32, 32);

    cudaError_t cudaErr = cudaSuccess;
    cudaStream_t stream;
    cudaErr = cudaStreamCreate(&stream);
    printCudaError(cudaErr, "cudaStreamCreate");

    cuda_unique_ptr<double> d_unique_ptr_error(cuda_new<double>(0), cuda_delete<double>);
    cuda_unique_ptr<void> d_unique_ptr_temp_storage(cuda_new<void>(0), cuda_delete<void>);
    

	double *d_error_ptr = d_unique_ptr_error.get();
	cudaErr = cudaMalloc((void**)&d_error_ptr, sizeof(double));
    printCudaError(cudaErr, "cudaMalloc");

    void *d_temp_storage = d_unique_ptr_temp_storage.get();
    size_t temp_storage_bytes = 0;
	//we call DeviceReduce here to check how much memory we need for temporary storage
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, Anew, d_error_ptr, m*m, stream);
    cudaErr = cudaMalloc(&d_temp_storage, temp_storage_bytes);
    printCudaError(cudaErr, "cudaMalloc");

    printf("temp_storage_bytes: %d\n\n", temp_storage_bytes);
    printf("Jacobi relaxation Calculation: %d x %d mesh\n", m, m);
    printf("Max iterations: %d\n", iter_max);
    printf("MIN Error: %lf\n\n", tol);
    auto start = std::chrono::high_resolution_clock::now();

    #pragma acc data copyin(A[:n*m], Anew[:n*m], Subtract_temp[:n*m])
    {
        int iter = 0;
        nvtxRangePushA("while");
        while (error > tol && iter < iter_max)
        {
            nvtxRangePushA("calcNext");
            #pragma acc parallel loop async(1) independent collapse(2) vector vector_length(n) gang num_gangs(n) present(A,Anew) 
            for (int j = 1; j < n - 1; j++){
                for (int i = 1; i < n - 1; i++){
                    Anew[OFFSET(j, i, n)] = 0.25 * (A[OFFSET(j, i + 1, n)] + A[OFFSET(j, i - 1, n)] + A[OFFSET(j - 1, i, n)] + A[OFFSET(j + 1, i, n)]);
                }
            }
            nvtxRangePop();
            if (iter % 100 == 0){
                nvtxRangePushA("calcError");
                error = 0;
                #pragma acc host_data use_device(A, Anew, Subtract_temp)
                    {
                        subtractArrays<<<grid, block, 0, stream>>>(A, Anew, Subtract_temp, m);
                        cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, Subtract_temp, d_error_ptr, m*m, stream);
                        cudaErr = cudaMemcpy(&error, d_error_ptr, sizeof(double), cudaMemcpyDeviceToHost);
                        printCudaError(cudaErr, "cudaMemcpy");
                    }
                nvtxRangePop();
            }

            nvtxRangePushA("swap");
            temp = A;
            A = Anew;
            Anew = temp;
            nvtxRangePop();

            if (iter % 1000 == 0)
                printf("%5d, %0.6f\n", iter, error);
            iter++;
        }
        nvtxRangePop();
        printf("%5d, %0.6f\n", iter, error);
        #pragma acc update host(A[:n*m], Anew[:n*m])
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;

    printf(" total: %f s\n", elapsed_seconds.count());
    std::ofstream out("out.txt");
    for (int j = 0; j < n; j++){
        for (int i = 0; i < m; i++){
            out << std::left << std::setw(10) << A[OFFSET(j, i, m)] << " ";
        }
        out << std::endl;
    }
    return 0;
}