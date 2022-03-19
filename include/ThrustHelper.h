#ifndef __THRUST_HELPER_H__
#define __THRUST_HELPER_H__

#include <opencv2/core/cuda.hpp>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

namespace ThrustHelper {
template <typename T>
struct step_functor : public thrust::unary_function<int, int> {
    int columns;
    int step;
    int channels;
    __host__ __device__ step_functor(int columns_, int step_, int channels_ = 1)
        : columns(columns_), step(step_), channels(channels_){};
    __host__ step_functor(cv::cuda::GpuMat &mat) {
        CV_Assert(mat.depth() == cv::DataType<T>::depth);
        columns = mat.cols;
        step = mat.step / sizeof(T);
        channels = mat.channels();
    }
    __host__ __device__ int operator()(int x) const {
        int row = x / columns;
        int idx = (row * step) + (x % columns) * channels;
        return idx;
    }
};

template <typename T>
thrust::permutation_iterator<
    thrust::device_ptr<T>,
    thrust::transform_iterator<step_functor<T>, thrust::counting_iterator<int>>>
GpuMatBeginItr(cv::cuda::GpuMat mat, int channel = 0) {
    if (channel == -1) {
        mat = mat.reshape(1);
        channel = 0;
    }
    CV_Assert(mat.depth() == cv::DataType<T>::depth);
    CV_Assert(channel < mat.channels());
    return thrust::make_permutation_iterator(
        thrust::device_pointer_cast(mat.ptr<T>(0) + channel),
        thrust::make_transform_iterator(
            thrust::make_counting_iterator(0),
            step_functor<T>(mat.cols, mat.step / sizeof(T), mat.channels())));
}

template <typename T>
thrust::permutation_iterator<
    thrust::device_ptr<T>,
    thrust::transform_iterator<step_functor<T>, thrust::counting_iterator<int>>>
GpuMatEndItr(cv::cuda::GpuMat mat, int channel = 0) {
    if (channel == -1) {
        mat = mat.reshape(1);
        channel = 0;
    }
    CV_Assert(mat.depth() == cv::DataType<T>::depth);
    CV_Assert(channel < mat.channels());
    return thrust::make_permutation_iterator(
        thrust::device_pointer_cast(mat.ptr<T>(0) + channel),
        thrust::make_transform_iterator(
            thrust::make_counting_iterator(mat.rows * mat.cols),
            step_functor<T>(mat.cols, mat.step / sizeof(T), mat.channels())));
}
} // namespace ThrustHelper
#endif