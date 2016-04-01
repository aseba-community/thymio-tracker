//class to define object model based on list of 3D edges
//edges are directed: when moves from pt1 to pt2 want black on the right

#pragma once

#include <iostream>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>

//for testing, to remove
//#include <opencv2/highgui/highgui.hpp>

#include "Generic.hpp"
#include "GH.hpp"
#include "Grouping.hpp"
#include "P3P.hpp"

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

struct ModelTriplet {
    int ids[3];
    ModelTriplet(){};
    ModelTriplet(int _id1,int _id2,int _id3){ids[0]=_id1;ids[1]=_id2;ids[2]=_id3;};
} ;
struct ModelQuadruplet {
    int ids[4];
    ModelQuadruplet(){};
    ModelQuadruplet(int _id1,int _id2,int _id3,int _id4){ids[0]=_id1;ids[1]=_id2;ids[2]=_id3;;ids[3]=_id4;};
} ;

struct PoseHypothesisSet {
    cv::Affine3d pose;
    float score;
    PoseHypothesisSet(){score=0;}
} ;

class Object3D
{
public:
    //constructor
    Object3D();
    ~Object3D();
    
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
    //do pose estimation from image using blob extraction, and voting P3P
    bool getPoseFromBlobs(const std::vector<cv::KeyPoint>& blobs,const IntrinsicCalibration& _mCalib, cv::Affine3d& robotPose, bool init);
    
    //3D model
    std::vector<cv::Point3f> mVertices;
    std::vector<ModelEdge> mEdges;
    
    //groups of vertices in our model
    std::vector<ModelTriplet> mGroup3s;
    std::vector<ModelQuadruplet> mGroup4s;
    
    //pose
    // TODO: is this used??
    cv::Affine3d pose;

    //information for time consistency
    int lengthHistory;
    int nbHypoPerTime;
    PoseHypothesisSet **PoseHypothesisHistory;//history x hypo (first item is most recent)

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

    cv::Mat mImage;
    std::vector<cv::Point2f> mRobotKeypointPos;
};

}
