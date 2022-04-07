#include <cstdio>
#include "Laplacian.cuh"

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf(
            "%s: Invalid number of command line arguments. Exiting program\n",
            argv[0]);
        printf("Usage: %s [image]", argv[0]);
    }

    cv::Mat h_oriImg = cv::imread(argv[1], cv::IMREAD_COLOR);

    Laplacian::gpuLaplacian(h_oriImg);

    Laplacian::cpuLaplacian(h_oriImg);

    return 0;
}
