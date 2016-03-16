//class to define object model based on list of 3D edges
//edges are directed: when moves from pt1 to pt2 want black on the right
#pragma once

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

#include <iostream>
#include <vector>

namespace thymio_tracker
{

std::vector<cv::Affine3d> computeP3P(const std::vector<cv::Point2f> &projected2DPoints,const std::vector<cv::Point3f> &Model3DPoints);


}
