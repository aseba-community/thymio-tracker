//class to define object model based on list of 3D edges
//edges are directed: when moves from pt1 to pt2 want black on the right

#pragma once

#include <iostream>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include "Generic.hpp"
#include "GH.hpp"

namespace thymio_tracker
{

// TODO: 
struct ModelEdge {
    cv::Point3f ptFrom;
    cv::Point3f ptTo;
    
    ModelEdge(const cv::Point3f& p1, const cv::Point3f& p2)
        : ptFrom(p1), ptTo(p2)
    {}
};

class Object3D
{
public:
    //constructor
    Object3D();
    
    // TODO: Remove all this from Object3D? Object3D should not draw itself.
    
    //draw object given camera intrinsic and extrinsic
    void draw(cv::Mat &img, const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs, const cv::Affine3d& poseCam) const;
    
    inline void draw(cv::Mat &img, const IntrinsicCalibration &_mCalib, const cv::Affine3d &poseCam) const
    {
        draw(img, _mCalib.cameraMatrix, _mCalib.distCoeffs, poseCam);
    }
    
    //draw object elements given camera intrinsic and extrinsic
    void drawVertice(const cv::Point3f &_vertice, cv::Mat &img, const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs, const cv::Affine3d &poseCam) const;
    void drawEdge(const ModelEdge &_edge, const cv::Mat &img, const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs, const cv::Affine3d &poseCam) const;
    //project vertices and return them in vector
    std::vector<cv::Point2f> projectVertices(const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs, const cv::Affine3d &poseCam) const;
    //do pose estimation using projection of vertices
    bool getPose(const IntrinsicCalibration& _mCalib, std::vector<DetectionGH> mMatches, cv::Affine3d& robotPose, bool init) const;
    
    //3D model
    std::vector<cv::Point3f> mVertices;
    std::vector<ModelEdge> mEdges;
    
    //pose
    // TODO: is this used??
    cv::Affine3d pose;
};

class Camera3dModel: public Object3D
{
public:
    //constructor
    Camera3dModel();
};

class ThymioBlobModel: public Object3D
{
public:
    //constructor
    ThymioBlobModel();
};

}
