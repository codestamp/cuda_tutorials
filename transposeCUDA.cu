/*Transpose Matrix (row,per element and tiled)
Autor: Munesh Singh
Date: 08 March 2010
*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"


const int N = 1024;		// matrix size is NxN
const int K = 32;	    // TODO, set K to the correct value and tile size will be KxK

__global__ void
transpose_parallel_tiled(float in[], float out[])
{
	// TODO
	__shared__ float tile[K][K];
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int tx = threadIdx.x; int ty = threadIdx.y;

	if (row < N && col < N) {
		tile[ty][tx] = in[row * N + col];
		__syncthreads();
		out[(col +ty -tx)*N + (row + tx - ty)] = tile[tx][ty];
		__syncthreads();
	}
}



// to be launched with one thread per element, in KxK threadblocks
// thread (x,y) in grid writes element (i,j) of output matrix 
__global__ void
transpose_parallel_per_element(float in[], float out[]) 
{
	//TODO
	int j = threadIdx.x + blockIdx.x * blockDim.x;
	int i = threadIdx.y + blockIdx.y * blockDim.y;
	if (j < N && i < N) {

		out[j + i * N] = in[i + j * N]; //out(j,i) = in(i,j)
	}
}

//The following functions and kernels are for your reference
void
transpose_CPU(float in[], float out[])
{
	for (int j = 0; j < N; j++)
		for (int i = 0; i < N; i++)
			out[j + i * N] = in[i + j * N]; // out(j,i) = in(i,j)
}

// to be launched on a single thread
__global__ void
transpose_serial(float in[], float out[])
{
	for (int j = 0; j < N; j++)
		for (int i = 0; i < N; i++)
			out[j + i * N] = in[i + j * N]; // out(j,i) = in(i,j)
}

// to be launched with one thread per row of output matrix
__global__ void
transpose_parallel_per_row(float in[], float out[])
{
	int i = threadIdx.x;

	for (int j = 0; j < N; j++)
		out[j + i * N] = in[i + j * N]; // out(j,i) = in(i,j)
}

void fill_matrix(float *in) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			in[i*N+j] = (float)(i*N+j);
		}
	}
}

int compare_matrices(float *out, float *gold) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			if (out[i * N + j] != gold[i * N + j]) {
				printf("out[%d]= %.f\tgold[%d]=%.f\n", i * N + j, out[i * N + j], i * N + j, gold[i * N + j]);
				printf("\n(%d,%d)\n", i, j);
				return 1;
			}
		}
	}
	return 0;
}

void print_matrices(float* a) {
	for (int i = 0; i < 5; i++) {
		printf("\n");
		for (int j = 0; j < 5; j++) {
			printf("%.f\t", a[i * N + j]);
		}
	}
	printf("\n");
}


int main(int argc, char** argv)
{
	int numbytes = N * N * sizeof(float);

	float* in = (float*)malloc(numbytes);
	float* out = (float*)malloc(numbytes);
	float* gold = (float*)malloc(numbytes);

	fill_matrix(in);
	transpose_CPU(in, gold);

	float* d_in, * d_out;

	cudaMalloc(&d_in, numbytes);
	cudaMalloc(&d_out, numbytes);
	cudaMemcpy(d_in, in, numbytes, cudaMemcpyHostToDevice);

	GpuTimer timer;

	/*
	 * Now time each kernel and verify that it produces the correct result.
	 *
	 * To be really careful about benchmarking purposes, we should run every kernel once
	 * to "warm" the system and avoid any compilation or code-caching effects, then run
	 * every kernel 10 or 100 times and average the timings to smooth out any variance.
	 * But this makes for messy code and our goal is teaching, not detailed benchmarking.
	 */

	dim3 blocks(N / K, N / K); // TODO, you need to define the correct blocks per grid
	dim3 threads(K, K);	// TODO, you need to define the correct threads per block

	timer.Start();
	//transpose_serial << <1, 1 >> > (d_in, d_out);
	//transpose_parallel_per_row << <1, N >> > (d_in, d_out);
	//transpose_parallel_per_element << <blocks, threads >> > (d_in, d_out);
	transpose_parallel_tiled << <blocks, threads >> > (d_in, d_out);
	timer.Stop();
	cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
	printf("\nIn matrix\n");
	print_matrices(in);
	printf("\nOut matrix\n");
	print_matrices(out);
	printf("\nGold matrix\n");
	print_matrices(gold);


	printf("transpose_parallel_per_element: %g ms.\nVerifying transpose...%s\n",
		timer.Elapsed(), compare_matrices(out, gold) ? "Failed" : "Success");

	cudaFree(d_in);
	cudaFree(d_out);
}
