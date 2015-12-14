
#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

#include <Vector>
#include <iostream>
#include <fstream>

namespace cv
{
    class SimpleBlobDetectorInertia : public cv::SimpleBlobDetector
    {
    public:
        //constructor
        void SimpleBlobDetectorImpl(){};
        CV_WRAP static Ptr<SimpleBlobDetectorInertia>
        create(const SimpleBlobDetectorInertia::Params &parameters = SimpleBlobDetectorInertia::Params());
    };
}
