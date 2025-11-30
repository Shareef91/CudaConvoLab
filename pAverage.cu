#include <cuda.h>
#include <stdio.h>

#define MAX_MASK_SIZE 10

__constant__ float MASK[MAX_MASK_SIZE];

__global__ void average_kernel(float *output,
                               float *input,
                               int input_size,
                               int mask_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x; // total number of threads running (used as stride)

    while (idx < input_size) { // process elements in a strided loop until all inputs handled
        float total = 0.0f; // running total for this output element
        int start = idx - mask_size + 1; // starting input index for the mask application
        for (int j = 0; j < mask_size; j++) { // iterate over mask elements
            int in_idx = start + j; // compute corresponding input index
            float input_value = in_idx >= 0 ? input[in_idx] : input[0]; // handle negative indices using input[0]
            total += input_value * MASK[j]; // accumulate weighted input
        }
        output[idx] = total; // write result to output array
        idx += stride; // move to the next element this thread should process
    }
}

void process_data(float *output,
                  float *input,
                  float *mask,
                  int input_size,
                  int mask_size) {
    float *d_input = NULL;
    float *d_output = NULL; // device pointer for output

    size_t bytes = input_size * sizeof(float); // number of bytes to copy for arrays

    cudaMalloc((void **)&d_input, bytes); // allocate device memory for input
    cudaMalloc((void **)&d_output, bytes); // allocate device memory for output

    cudaMemcpy(d_input, input, bytes, cudaMemcpyHostToDevice); // copy input array to device

    cudaMemcpyToSymbol(MASK,                       // copy mask into constant memory symbol
                       mask,
                       mask_size * sizeof(float),
                       0,
                       cudaMemcpyHostToDevice);

    int threads = 256; // threads per block
    int blocks = 256; // number of blocks (chosen; can be tuned)
    if (blocks == 0) blocks = 1; // safeguard (shouldn't happen here)

    average_kernel<<<blocks, threads>>>(d_output, d_input, input_size, mask_size); // launch kernel
    cudaDeviceSynchronize(); // wait for kernel to finish

    cudaMemcpy(output, d_output, bytes, cudaMemcpyDeviceToHost); // copy results back to host

    cudaFree(d_input); // free device input memory
    cudaFree(d_output); // free device output memory
}

int main(int argc, char **argv) {
    FILE *infile;
    FILE *outfile;

    float *input;
    float *output;
    float mask[] = {0.1, 0.2, 0.3, 0.4};

    int i;
    int n;

    if (argc < 3) {
        fprintf(stderr, "Usage: %s <infile> <outfile>\n", argv[0]);
        exit(1);
    }

    infile = fopen(argv[1], "r");
    if (infile == NULL) {
        fprintf(stderr, "Error: cannot open input file [%s].\n", argv[1]);
        exit(1);
    }

    fscanf(infile, "%d", &n);
    input = (float *) malloc(n * sizeof(float));

    for (i = 0; i < n; i++) {
        fscanf(infile, "%f", &(input[i]));
    }

    fclose(infile);

    output = (float *) malloc(n * sizeof(float));
    process_data(output, input, mask, n, 4);

    outfile = fopen(argv[2], "w");
    fprintf(outfile, "%d\n", n);

    for (i = 0; i < n; i++) {
        fprintf(outfile, "%.3f\n", output[i]);
    }

    fclose(outfile);

    free(input);
    free(output);

    return 0;
}

// -------------------------------------------------------------------------
// AI Usage Disclosure
// Date: 2025-11-30
// Service: ChatGPT (GPT-5.1 Thinking)
// Purpose: I wrote the CUDA code (average_kernel and process_data) myself.
//          I only used ChatGPT to help me check my work, explain what my
//          code is doing, and remind me of the correct nvcc commands to
//          compile and run the program on my machine.
// -------------------------------------------------------------------------
