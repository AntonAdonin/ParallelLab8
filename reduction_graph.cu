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
    if ((i >= 0) && (i < m) && (j >= 0) && (j < m)) {
        Sub_res[OFFSET(i,j,m)] = fabs(A[OFFSET(i,j,m)] - Anew[OFFSET(i,j,m)]);
    }
}


__global__ void getAverage(double *A, double *Anew, int m, bool calcLeft) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (calcLeft){
        if ((i > 0) && (i < m - 1) && (j > 0) && (j < m - 1)) {
            A[OFFSET(j, i, m)] = 0.25 * (Anew[OFFSET(j, i + 1, m)] + Anew[OFFSET(j, i - 1, m)]
            + Anew[OFFSET(j - 1, i, m)] + Anew[OFFSET(j + 1, i, m)]);
        }
    }else{
        if ((i > 0) && (i < m - 1) && (j > 0) && (j < m - 1)) {
            Anew[OFFSET(j, i, m)] = 0.25 * (A[OFFSET(j, i + 1, m)] + A[OFFSET(j, i - 1, m)]
            + A[OFFSET(j - 1, i, m)] + A[OFFSET(j + 1, i, m)]);
        }
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

    // размерности grid и block
    dim3 grid(32 , 32);
	dim3 block(32, 32);

    cudaError_t cudaErr = cudaSuccess;
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cuda_unique_ptr<double> d_unique_ptr_error(cuda_new<double>(0), cuda_delete<double>);
    cuda_unique_ptr<void> d_unique_ptr_temp_storage(cuda_new<void>(0), cuda_delete<void>);

    cuda_unique_ptr<double> d_unique_ptr_A(cuda_new<double>(0), cuda_delete<double>);
    cuda_unique_ptr<double> d_unique_ptr_Anew(cuda_new<double>(0), cuda_delete<double>);
    cuda_unique_ptr<double> d_unique_ptr_Subtract_temp(cuda_new<double>(0), cuda_delete<double>);
    
    // выделение памяти и перенос на GPU
	double *d_error_ptr = d_unique_ptr_error.get();
	cudaErr = cudaMalloc((void**)&d_error_ptr, sizeof(double));
    printCudaError(cudaErr, "cudaMalloc");

    double *d_A = d_unique_ptr_A.get();
  	cudaErr = cudaMalloc((void **)&d_A, m*m*sizeof(double));
    printCudaError(cudaErr, "cudaMalloc");

	double *d_Anew = d_unique_ptr_Anew.get();
	cudaErr = cudaMalloc((void **)&d_Anew, m*m*sizeof(double));
    printCudaError(cudaErr, "cudaMalloc");

    double *d_Subtract_temp = d_unique_ptr_Subtract_temp.get();
	cudaErr = cudaMalloc((void **)&d_Subtract_temp, m*m*sizeof(double));
    printCudaError(cudaErr, "cudaMalloc");

    cudaErr = cudaMemcpy(d_A, A, m*m*sizeof(double), cudaMemcpyHostToDevice);
    printCudaError(cudaErr, "cudaMemcpy");

    cudaErr = cudaMemcpy(d_Anew, Anew, m*m*sizeof(double), cudaMemcpyHostToDevice);
    printCudaError(cudaErr, "cudaMemcpy");
	// проверка занимаемой памяти для редукции
    void *d_temp_storage = d_unique_ptr_temp_storage.get();
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_Anew, d_error_ptr, m*m, stream);
    cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);

    printf("temp_storage_bytes: %d\n", temp_storage_bytes);
    printf("Jacobi relaxation Calculation: %d x %d mesh\n", m, m);
    printf("Max iterations: %d\n", iter_max);
    printf("MIN Error: %lf\n\n", tol);

    // graph
    bool graph_created = false;
	cudaGraph_t graph;
	cudaGraphExec_t instance;

    int iter = 0;
    auto start = std::chrono::high_resolution_clock::now();
    
    cudaDeviceSynchronize();
    nvtxRangePushA("while");
    while (error > tol && iter < iter_max)
    {
        if(!graph_created) {
            // создание графа
            nvtxRangePushA("createGraph");
            cudaErr = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
            printCudaError(cudaErr, "cudaStreamBeginCapture");
            for (int i = 0; i < 100; i++) {
                getAverage<<<grid, block, 0, stream>>>(d_A, d_Anew, m, (bool)(i % 2));
            }
            cudaErr = cudaStreamEndCapture(stream, &graph);
            printCudaError(cudaErr, "cudaStreamEndCapture");
            nvtxRangePop();
            cudaErr = cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
            printCudaError(cudaErr, "cudaGraphInstantiate");
            graph_created = true;
        }
        nvtxRangePushA("startGraph");
        //запуск графа
        cudaErr = cudaGraphLaunch(instance, stream);
        printCudaError(cudaErr, "cudaGraphLaunch");
        
        nvtxRangePop();
        iter += 100;
        if (iter % 100 == 0){
            nvtxRangePushA("calcError");
            subtractArrays<<<grid, block, 0, stream>>>(d_A, d_Anew, d_Subtract_temp, m);
            cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_Subtract_temp, d_error_ptr, m*m, stream);
            cudaErr = cudaMemcpy(&error, d_error_ptr, sizeof(double), cudaMemcpyDeviceToHost);
            printCudaError(cudaErr, "cudaMemcpy");
            nvtxRangePop();
        }
        if (iter % 1000 == 0)
            printf("%5d, %0.6f\n", iter, error);
    }
    nvtxRangePop();
    printf("%5d, %0.6f\n", iter, error);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;

    printf("total: %f s\n", elapsed_seconds.count());
    cudaErr = cudaMemcpy(A, d_A, m*m*sizeof(double), cudaMemcpyDeviceToHost);
    printCudaError(cudaErr, "cudaMemcpy");
    std::ofstream out("out.txt");
    for (int j = 0; j < n; j++){
        for (int i = 0; i < m; i++){
            out << std::left << std::setw(10) << A[OFFSET(j, i, m)] << " ";
        }
        out << std::endl;
    }
    return 0;
}