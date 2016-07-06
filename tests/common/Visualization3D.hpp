//Viewer object, to visualize 3D objects in a sort of doom like fashion.

#pragma once

#include <iostream>
#include <vector>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>

#include "Generic.hpp"
#include "Models.hpp"

namespace tt = thymio_tracker;

class Visualization3D
{
public:
    //constructor
    Visualization3D(tt::IntrinsicCalibration *_mPtrCalibration);
    ~Visualization3D();
    
    void addObject(tt::Object3D &_obj);
    void draw();
    
    //3D objects
    std::vector<tt::Object3D> mObjects;
    
    //name window
    std::string windowName;
    //background
    cv::Mat imBackground;
    cv::Mat imCurrent;
    //camera calibration of viewer
    tt::IntrinsicCalibration *mPtrCalibration;
    //current position of camera
    cv::Affine3d cameraPose;

    
};