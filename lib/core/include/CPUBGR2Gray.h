#ifndef __CPU_BGR2GRAY_H__
#define __CPU_BGR2GRAY_H__

#include <opencv4/opencv2/core/mat.hpp>

namespace CPU {
template <typename T_in, typename T_out>
void BGR2Gray(const cv::Mat &src, cv::Mat &dst, size_t cols, size_t rows) {
    for (size_t i = 0; i < rows; i++) {
        auto srcPtr = src.ptr<T_in>(i);
        for (size_t j = 0; j < cols; j++) {
            auto t = srcPtr[j];
            dst.at<T_out>(i, j) =
                (T_out)0.114f * t.x + 0.587f * t.y + 0.299f * t.z;
        }
    }
}
} // namespace CPU

#endif
