/* 
Autor: Munesh Singh
Date: 08 March 2010
Vector addition using cudaMallocPitch
*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

const int width = 567;
const int height = 985;

__global__ void testKernel2D(float* M, float* N, float* P, size_t pitch) {
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	if (row < width && col < width) {
		float* row_M = (float*)((char*)M + row * pitch);
		float* row_N = (float*)((char*)N + row * pitch);
		float* row_P = (float*)((char*)P + row * pitch);

		row_P[col] = row_M[col] + row_N[col];
	}
}

void printMatrix(float* K) {
	for (int i = 110; i < 116; i++) {
		printf("\n");
		for (int j = 213; j < 219; j++) {
			printf("%.3f\t", K[i * width + j]);
		}
	}
}

int main() {
	srand(time(NULL));
	float* M, * N, * P;
	size_t size = sizeof(float) * width * height;
	int hpitch = sizeof(float) * width;
	M = (float*)malloc(size);
	N = (float*)malloc(size);
	P = (float*)malloc(size);

	//cuda timer
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);


	for (int i = 0; i < width * height; i++) {
		M[i] = 1 + static_cast<float> (rand()) / (static_cast<float>(RAND_MAX / (20 - 1)));
		N[i] = 1 + static_cast<float> (rand()) / (static_cast<float>(RAND_MAX / (20 - 1)));
	}

	float* dM, * dN, * dP;
	size_t dpitch;
	cudaMallocPitch(&dM, &dpitch, width * sizeof(float), height);
	cudaMallocPitch(&dN, &dpitch, width * sizeof(float), height);
	cudaMallocPitch(&dP, &dpitch, width * sizeof(float), height);

	cudaMemcpy2D(dM, dpitch, M, hpitch, width * sizeof(float), height, cudaMemcpyHostToDevice);
	cudaMemcpy2D(dN, dpitch, N, hpitch, width * sizeof(float), height, cudaMemcpyHostToDevice);

#define idiv(a,n) (a%n?a/n+1:a/n)

	dim3 nThreads(32, 32, 1);
	dim3 nBlocks(idiv(width, nThreads.x), idiv(height, nThreads.y), 1);

	cudaEventRecord(start);
	testKernel2D << <nBlocks, nThreads >> > (dM, dN, dP, dpitch);
	//cudaDeviceSynchronize();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaMemcpy2D(P, hpitch, dP, dpitch, width * sizeof(float), height, cudaMemcpyDeviceToHost);

	printf("Matrix M:\n");
	printMatrix(M);
	printf("\n\nMatrix N:\n");
	printMatrix(N);
	printf("\n\nMatrix P:\n");
	printMatrix(P);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("\n\nElapsed time in millis: %.4f\n", elapsedTime);
	printf("\nDevice pitch (calculated by cudaMallocPitch): %zu\n", dpitch);
	cudaFree(dM);
	cudaFree(dN);
	cudaFree(dP);

	return 0;
}

