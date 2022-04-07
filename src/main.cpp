#include <cstdio>
#include "Laplacian.cuh"

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf(
            "%s: Invalid number of command line arguments. Exiting program\n",
            argv[0]);
        printf("Usage: %s [image]", argv[0]);
    }

    Mat h_oriImg = imread(argv[1], IMREAD_COLOR);

    gpuLaplacian(h_oriImg);

    cpuLaplacian(h_oriImg);

    return 0;
}
