//class to define object model based on list of 3D edges
//edges are directed: when moves from pt1 to pt2 want black on the right
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
    Visualization3D();
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
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
    //current position of camera
    cv::Affine3d cameraPose;

    
};