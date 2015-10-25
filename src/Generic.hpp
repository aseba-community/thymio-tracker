//class to define object model based on list of 3D edges
//edges are directed: when moves from pt1 to pt2 want black on the right
#pragma once

#include <opencv2/core/core.hpp>

#include <iostream>
#include <vector>

struct IntrinsicCalibration {
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
} ;

//get camera intrisic matrix in case of resize of image
void resizeCameraMatrix(cv::Mat &_cameraMatrix, const cv::Size& sFrom, const cv::Size& sTo);

//pixel to meter transfo and inverse
cv::Point2f toPixels(const cv::Mat& _cameraMatrix, const cv::Point2f& _m);
cv::Point2f toMeters(const cv::Mat& _cameraMatrix, const cv::Point2f& _x);
//project to image plane
cv::Point2f toImagePlane(const cv::Point3f& _x);
//get jacobian of point projection with respect to camera position
cv::Mat ProjectZ1_Jac_Dp(const cv::Point3f& mvLastDistCam);
