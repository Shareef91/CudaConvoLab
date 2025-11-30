#include <cuda.h>
#include <stdio.h>

void process_data(float *output,
                  float *input,
                  float *mask,
                  int input_size,
                  int mask_size) {
    int i;
    int j;

    for (i = 0; i < input_size; i++) {
        float total = 0.0;
        int start = i - mask_size + 1;
        for (j = 0; j < mask_size; j++) {
            float input_value = start + j >= 0 ? input[start + j] : input[0];
            total += input_value * mask[j];
        }
        output[i] = total;
    }
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
