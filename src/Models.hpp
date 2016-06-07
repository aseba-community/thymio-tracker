//class to define object model based on list of 3D edges
//edges are directed: when moves from pt1 to pt2 want black on the right

#pragma once

#include <iostream>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>

//for testing, to remove
//#include <opencv2/highgui/highgui.hpp>

#include "Generic.hpp"
#include "GH.hpp"
#include "Grouping.hpp"
#include "TrackingFcts.hpp"

namespace thymio_tracker
{

//edges just for plotting
struct ModelEdge {
    cv::Point3f ptFrom;
    cv::Point3f ptTo;
    
    ModelEdge(const cv::Point3f& p1, const cv::Point3f& p2)
        : ptFrom(p1), ptTo(p2)
    {}
};

//edge structure for tracking: need also sort of normal, ie vector which defines the bissectrice of the inner structure
//and radius of edge rounding ?
struct ModelEdgeTrack {
    cv::Point3f ptFrom;
    cv::Point3f ptTo;

    //cv::Vec3d normal;
    cv::Vec3d normal1;//normal first plane pointing inward
    cv::Vec3d normal2;//normal second plane
    float rounding;
    
    //ModelEdgeTrack(const cv::Point3f& p1, const cv::Point3f& p2, const cv::Vec3d& n, const float& r = 0)
    //    : ptFrom(p1), ptTo(p2), normal(n), rounding(r)
    ModelEdgeTrack(const cv::Point3f& p1, const cv::Point3f& p2, const cv::Vec3d& n1, const cv::Vec3d& n2, const float& r = 0)
        : ptFrom(p1), ptTo(p2), normal1(n1), normal2(n2), rounding(r)
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

#define surfacePpmm 2. // pixel per millimeter fir appearance definition in planarSurface

struct planarSurface {
    cv::Point3f center;
    //basis vector
    cv::Vec3d b1;
    cv::Vec3d b2;
    cv::Vec3d normal;

    float radius1;
    float radius2;

    cv::Mat mImage;
    //information to learn appearance of face
    float weight; //defined by accumulation of weights of each appearances accumulated in mImageWrite (depending on angle view)

    planarSurface(const cv::Point3f _center, const cv::Vec3d _b1, const cv::Vec3d _b2, const float _radius)
        : center(_center), b1(_b1), b2(_b2), radius1(_radius), radius2(_radius)    {normal = b1.cross(b2); weight=0;}
    planarSurface(const cv::Point3f _center, const cv::Vec3d _b1, const cv::Vec3d _b2, const float _radius1, const float _radius2)
        : center(_center), b1(_b1), b2(_b2), radius1(_radius1), radius2(_radius2)    {normal = b1.cross(b2); weight=0; }

    void allocateLearning()
    {
        weight=0;
        mImage.create((int)(1000.*radius2*surfacePpmm),(int)(1000.*radius1*surfacePpmm), CV_32FC1);
        mImage.setTo(0.);
    }
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
    void drawEdgeTrack(const ModelEdgeTrack &_edge, const cv::Mat &img, const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs, const cv::Affine3d &poseCam) const;
    void drawSurface(const planarSurface &_edge, const cv::Mat &img, const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs, const cv::Affine3d &poseCam) const;
    
    //project vertices and return them in vector
    //std::vector<cv::Point2f> projectVertices(const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs, const cv::Affine3d &poseCam) const;
    //do pose estimation using projection of vertices and matches from GH
    //bruteforce pose estimation: all subsets of 4 matches are used to solve PnP until half of matches agree
    bool getPose(const IntrinsicCalibration& _mCalib, std::vector<DetectionGH> mMatches, cv::Affine3d& robotPose, bool init) const;
    
    //3D model
    std::vector<cv::Point3f> mVertices;
    std::vector<ModelEdge> mEdges;
    std::vector<ModelEdgeTrack> mEdgesTrack;

    //texture information from piece wise planar areas
    std::vector<planarSurface> mPlanarSurfaces;

    //allocate float grayscale iamge for each planar surface where to accumulate appearance defined from tracking board
    void allocateSurfaceLearning();
    //use an image and corresponding robot pose to update surface appearances
    void learnAppearance(cv::Mat &img, const IntrinsicCalibration &_mCalib, const cv::Affine3d& poseCam);
    //all board sequence has been used => save appearance
    void writeSurfaceLearned();
    void readSurfaceLearned();

    //temporary tracking function for develop, will have to be moved to Robot if works
    //void track(const cv::Mat &img, const IntrinsicCalibration &_mCalib, const cv::Affine3d& prevPoseCam, cv::Affine3d& poseCam) const;
    void track(const cv::Mat &img, const cv::Mat &prev_img, const IntrinsicCalibration &_mCalib, const cv::Affine3d& prevPoseCam, cv::Affine3d& poseCam) const;

    
    //groups of vertices in our model
    std::vector<ModelTriplet> mGroup3s;
    std::vector<ModelQuadruplet> mGroup4s;
    
    //pose
    // TODO: is this used??
    cv::Affine3d pose;

    //information for time consistency
    /*int lengthHistory;
    int nbHypoPerTime;
    PoseHypothesisSet **PoseHypothesisHistory;//history x hypo (first item is most recent)*/

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
    void setBlobModel();
    void setEdgePlotModel();
    void setEdgeTrackModel();
    void setSurfacesModel();
    void loadTrackingModel(cv::FileStorage& robotModelStorage);

    //for tracking using top surface and similar approach as landmark => not robust enough
    cv::Mat mImage;
    std::vector<cv::Point2f> mRobotKeypointPos;
    std::vector<cv::Point2f> mVerticesTopPos;
};

}
