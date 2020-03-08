/* Squarematrix multiply
Author: Munesh Singh
Date: 08 March 2020
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "timer.h"

const int N = 1024; 
const int K = 32;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

void compare_matrices(int* a, int* b) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			assert("Not matching!" && a[i * N + j] == b[i * N + j]);
		}
	}
	printf("Matrices match!");
}


void print_matrix(int* a) {
	for (int i = 0; i < 10; i++) {
		printf("\n");
		for (int j = 0; j < 10; j++) {
			printf("%d\t", a[i * N + j]);
		}
	}
	printf("\n");
}

void fill_matrix(int* a) {
	for (int i = 0; i < N * N; i++)
		a[i] = rand() % 10;
}

//34.298 ms
__global__
void gpu_matrix_multiply_tiled(int* a, int* b, int* c) {
	__shared__ int Ms[K][K];
	__shared__ int Ns[K][K];
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int sum = 0;
	for (int p = 0; p < N / K; p++) {
		Ms[ty][tx] = a[row * N + p * K + tx];
		Ns[ty][tx] = b[(p * K + ty) * N + col];
		__syncthreads();

		for (int k = 0; k < K; k++) {
			sum += Ms[ty][k] * Ns[k][tx];
		}
		__syncthreads();
	}
		c[row * N + col] = sum;
}

//1316.538 ms
__global__
void gpu_matrix_multiply_per_row(int* a, int* b, int* c) {
	size_t tx = threadIdx.x;
	
	if (tx < N) {
		
		for (int j = 0; j < N; j++) {
			int sum = 0;
			for (int k = 0; k < N; k++) {
				sum += a[tx * N + k] * b[k * N + j];
			}
			c[tx * N + j] = sum;
		}
	}
}

// 72.786 ms
__global__
void gpu_matrix_multiply_per_element(int* a, int* b, int* c) {
	size_t col = threadIdx.x + blockIdx.x * blockDim.x;
	size_t row = threadIdx.y + blockIdx.y * blockDim.y;
	if (row < N && col < N) {
		int sum = 0;
		for (int k = 0; k < N; k++) {
			sum += a[row * N + k] * b[k * N + col];
		}
		c[row * N + col] = sum;
	}
}

__global__
void gpu_matrix_multiply_serial(int* a, int* b, int* c) {
	/* for larger values of N, for example 128 or above this kernel gives an exception */
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < N; k++) {
				c[i * N + j] += a[i * N + k] * b[k * N + j];
			}
		}
	}
}

void cpu_matrix_multiply(int* a, int* b, int* c) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < N; k++) {
				c[i * N + j] += a[i * N + k] * b[k * N + j];
			}
		}
	}
}

int main() {
	int* ha, * hb, * gold, * hc;
	size_t size = N * N * sizeof(int);
	GpuTimer timer;

	ha = (int*)malloc(size);
	hb = (int*)malloc(size);
	hc = (int*)malloc(size);
	gold = (int*)malloc(size);

	fill_matrix(ha);
	fill_matrix(hb);
	memset(gold, 0, size);

	cpu_matrix_multiply(ha, hb, gold);

	int* da, * db, * dc;
	gpuErrchk(cudaMalloc(&da, size));
	gpuErrchk(cudaMalloc(&db, size));
	gpuErrchk(cudaMalloc(&dc, size));


	gpuErrchk(cudaMemcpy(da, ha, size, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(db, hb, size, cudaMemcpyHostToDevice));

	dim3 nThreads(K, K, 1);
	dim3 nBlocks((N + nThreads.x - 1) / nThreads.x, (N + nThreads.y - 1) / nThreads.y);

	timer.Start();
	
	//gpu_matrix_multiply_serial << <1, 1 >> > (da, db, dc);
	//gpu_matrix_multiply_per_element << <nBlocks, nThreads >> > (da, db, dc);
	//gpu_matrix_multiply_per_row << <1, N >> > (da, db, dc);
	gpu_matrix_multiply_tiled << <nBlocks, nThreads >> > (da, db, dc);
	
	timer.Stop();
	
	gpuErrchk(cudaMemcpy(hc, dc, size, cudaMemcpyDeviceToHost));

	/*
	printf("\nA matrix\n");
	print_matrix(ha);
	printf("\nB matrix\n");
	print_matrix(hb);
	printf("\nGold matrix\n");
	print_matrix(gold);
	printf("\nGPU result matrix\n");
	print_matrix(hc);
	*/
	printf("\n");
	printf("\nElapsed time for kernel \"gpu_matrix_multiply_serial\": %.3f milli-seconds\n", timer.Elapsed());
	compare_matrices(hc, gold);

	free(ha);
	free(hb);
	free(gold);

	gpuErrchk(cudaFree(da));
	gpuErrchk(cudaFree(db));
	gpuErrchk(cudaFree(dc));

	return 0;
}
	  
