//class to define object model based on list of 3D edges
//edges are directed: when moves from pt1 to pt2 want black on the right
#pragma once

#include <opencv2/core.hpp>

#include <iostream>
#include <vector>

namespace thymio_tracker
{

struct IntrinsicCalibration {
    cv::Size imageSize;
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
    IntrinsicCalibration(){};
} ;

struct DetectionGH {
    cv::Point2f position;//blob/feature position
    int id;//estimated id from GH
    float nbVotes;//nb votes for id
    int discriminativePower;//nb votes best - nb votes second best
    DetectionGH(cv::Point2f _p, int _id, int _nv, int _d): position(_p), id(_id), nbVotes(_nv), discriminativePower(_d) {};
} ;

//get the calibration from an open file
void readCalibrationFromFileStorage(cv::FileStorage &fs, IntrinsicCalibration &calib);
void writeCalibrationToFileStorage(IntrinsicCalibration &calib,cv::FileStorage &fs);

//get camera intrisic matrix in case of resize of image
void resizeCameraMatrix(cv::Mat &_cameraMatrix, const cv::Size& sFrom, const cv::Size& sTo);
void rescaleCalibration(IntrinsicCalibration &calibration, const cv::Size& sTo);

//pixel to meter transfo and inverse
cv::Point2f toPixels(const cv::Mat& _cameraMatrix, const cv::Point2f& _m);
cv::Point2f toMeters(const cv::Mat& _cameraMatrix, const cv::Point2f& _x);
cv::Point2f toMeters(const cv::Mat& _cameraMatrix, const cv::Point2i& _x);
//project to image plane
cv::Point2f toImagePlane(const cv::Point3f& _x);
//get jacobian of point projection with respect to camera position
cv::Mat ProjectZ1_Jac_Dp(const cv::Point3f& mvLastDistCam);
//get x,y coordinates of Point3f
cv::Point2f Pointxy(const cv::Point3f& _m);

//check direction triangle
bool testDirectionBasis(cv::Point2f basis1,cv::Point2f basis2);
bool testDirectionGroup(cv::Point2f v1,cv::Point2f v2,cv::Point2f v3);

}
