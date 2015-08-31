
#ifndef _ARTHYMIO_H
#define _ARTHYMIO_H

#include <opencv2/core/core.hpp>

int process(const cv::Mat& input, cv::Mat& output);

int get_rows(const cv::Mat& m);

#endif // _ARTHYMIO_H
