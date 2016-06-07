//class to define object model based on list of 3D edges
//edges are directed: when moves from pt1 to pt2 want black on the right
#pragma once

#include <opencv2/core.hpp>

#include <iostream>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>


namespace thymio_tracker
{

//robustPnp: solve a PnP problem iteratively using Levenberg Marquardt
//same as opencv solvePnp expect that the least squared error is weighted
//using the list of weights stored in score parameter
bool robustPnp(cv::InputArray opoints,cv::InputArray ipoints,
    cv::InputArray score, cv::InputArray _cameraMatrix, cv::InputArray _distCoeffs,
               cv::OutputArray _rvec, cv::OutputArray _tvec);

//same as matchTemplate from opencv except that the similarity function computed
//is the mutual information which has a more sharp optimum and is more robust to 
//occlusions and illumination variation
void matchTemplateMI( cv::Mat img, cv::Mat &templ, cv::Mat &res, cv::Mat &mask);

//function to refine the result provided by the minMaxLoc opencv function which
//provides maxLoc the pixelic location with the maximum score in an image of float curv
//To do so use the 4 closest neigbors to do parabic fitting and refine location
void parabolicRefinement(cv::Mat &curv,cv::Point maxLoc,cv::Point2f &maxLocF);


}
