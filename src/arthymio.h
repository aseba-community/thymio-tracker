
#ifndef _ARTHYMIO_H
#define _ARTHYMIO_H

#include <opencv2/core/core.hpp>

int process(const cv::Mat& input, cv::Mat& output,
            const cv::Mat* deviceOrientation=0);

#endif // _ARTHYMIO_H
