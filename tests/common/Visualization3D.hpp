//class to define object model based on list of 3D edges
//edges are directed: when moves from pt1 to pt2 want black on the right
#pragma once

#include <iostream>
#include <Vector>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "opencv2/highgui/highgui.hpp"

#include "Generic.hpp"
#include "Models.hpp"

using namespace std;
using namespace cv;


class Visualization3D
{
public:
    //constructor
    Visualization3D();
    ~Visualization3D();
    
    void addObject(Object3D &_obj);
    void draw();
    
    //3D objects
    vector<Object3D> mObjects;
    
    //name window
    string windowName;
    //background
    Mat imBackground;
    Mat imCurrent;
    //camera calibration of viewer
    Mat cameraMatrix;
    Mat distCoeffs;
    //current position of camera
    Affine3d cameraPose;

    
};