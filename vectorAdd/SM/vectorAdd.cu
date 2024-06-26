/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>
#include <stdlib.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */

__device__ unsigned int get_smid(void) {
    unsigned int ret;
    asm("mov.u32 %0, %smid;" : "=r"(ret) );
    return ret;
}


__global__ void
vectorAdd(const float *A, const float *B, float *C, int numElements, int size, int *SM_ids)
{
    int smid = get_smid();

    for (int j=0; j < size; j++){
        if (smid == SM_ids[j]){
            for (int i = threadIdx.x+blockDim.x*j; i < numElements; i += size*blockDim.x)
            {
                C[i] = A[i] + B[i];
            }
        }
    }
}

/**
 * Host main routine
 */
int main(int argc, char *argv[])
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    float elapsed_time;
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);

    // Print the vector length to be used, and compute its size
    int numElements = 5000000;
    int threadsPerBlock = 256;
    int blocksPerGrid = 80;
    int coreMax = 1;

    // Check if command line arguments were provided
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-size") == 0 && i + 1 < argc) {
            // Convert the next argument to an integer and use it as the size
            numElements = atoi(argv[i + 1]);
        }
        else if (strcmp(argv[i], "-threads") == 0 && i + 1 < argc) {
            // Convert the next argument to an integer and use it as the number of threads per block
            threadsPerBlock = atoi(argv[i + 1]);
        }
        else if (strcmp(argv[i], "-blocks") == 0 && i + 1 < argc) {
            // Convert the next argument to an integer and use it as the number of blocks per grid
            blocksPerGrid = atoi(argv[i + 1]);
        }
        else if (strcmp(argv[i], "-coremax") == 0 && i + 1 < argc) {
            // Convert the next argument to an integer and use it as the maximum core number
            coreMax = atoi(argv[i + 1]);
        }
    }

    // Allocate the hSM_ids array and fill it with numbers from 0 to coreMax
    int *hSM_ids = (int *)malloc(sizeof(int) * (coreMax));
    for (int i = 0; i < coreMax; i++) {
        hSM_ids[i] = i;
    }

    // Allocate the dSM_ids array and copy the hSM_ids array to the device
    int *dSM_ids;
    cudaMalloc((void**)&dSM_ids, sizeof(int) * (coreMax ));
    cudaMemcpy(dSM_ids, hSM_ids, sizeof(int) * (coreMax ), cudaMemcpyHostToDevice);

    size_t size = numElements * sizeof(float);
    //printf("[Vector addition of %d elements]\n", numElements);

    // Allocate the host input vector A
    float *h_A = (float *)malloc(size);

    // Allocate the host input vector B
    float *h_B = (float *)malloc(size);

    // Allocate the host output vector C
    float *h_C = (float *)malloc(size);

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    // Allocate the device input vector A
    float *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector B
    float *d_B = NULL;
    err = cudaMalloc((void **)&d_B, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector C
    float *d_C = NULL;
    err = cudaMalloc((void **)&d_C, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    //printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the Vector Add CUDA Kernel

    //printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    long flops = 2 * numElements;
    int repeat = 100;

    cudaEventRecord(beg);
    for (int i=0; i<repeat; i++){
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements, coreMax, dSM_ids);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, beg, end);
    elapsed_time /= 1000.; // Convert to seconds
    err = cudaGetLastError();

    printf(
        "%7.1f\n", //GFLOPS
        (repeat * flops * 1e-9) / elapsed_time);
    fflush(stdout);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    //printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Verify that the result vector is correct
    for (int i = 0; i < numElements; ++i)
    {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    //printf("Test PASSED\n");

    // Free device global memory
    err = cudaFree(d_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_B);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_C);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //printf("Done\n");
    return 0;
}

