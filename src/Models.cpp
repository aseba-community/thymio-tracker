#include "Models.hpp"
#include <stdexcept>


using namespace cv;
using namespace std;

namespace thymio_tracker
{

Object3D::Object3D()
{
    /*lengthHistory = 4;
    nbHypoPerTime = 4;
    PoseHypothesisHistory = new PoseHypothesisSet*[lengthHistory];
    for(int i=0;i<nbHypoPerTime;i++)
        PoseHypothesisHistory[i] = new PoseHypothesisSet[nbHypoPerTime];*/
}
Object3D::~Object3D()
{
    /*for(int i=0;i<nbHypoPerTime;i++)
        delete[] PoseHypothesisHistory[i];
    delete[] PoseHypothesisHistory;*/
}

void Object3D::draw(Mat& img, const Mat& cameraMatrix, const Mat& distCoeffs, const Affine3d& poseCam) const
{
    for(unsigned int e=0;e<mEdges.size();e++)
        drawEdge(mEdges[e], img, cameraMatrix, distCoeffs, poseCam);
    
    for(unsigned int v=0;v<mVertices.size();v++)
        drawVertice(mVertices[v], img, cameraMatrix, distCoeffs, poseCam);
    
    //draw object frame (axis XYZ)
    vector<Point3f> framePoints;
    framePoints.push_back(Point3f(0,0,0));
    framePoints.push_back(Point3f(0.03,0,0));
    framePoints.push_back(Point3f(0,0.03,0));
    framePoints.push_back(Point3f(0,0,0.03));

    for(int p=0;p<4;p++)framePoints[p]=pose*framePoints[p];
    vector<Point2f> vprojVertices;
    projectPoints(framePoints, poseCam.rvec(), poseCam.translation(), cameraMatrix, distCoeffs, vprojVertices);
    line(img, vprojVertices[0], vprojVertices[1], Scalar(0,0,255), 2);
    line(img, vprojVertices[0], vprojVertices[2], Scalar(0,255,0), 2);
    line(img, vprojVertices[0], vprojVertices[3], Scalar(255,0,0), 2);

}

vector<Point2f> Object3D::projectVertices(const Mat &cameraMatrix, const Mat &distCoeffs, const Affine3d &poseCam) const
{
    vector<Point2f> vProjPoints;
    Affine3d poseComb=poseCam * pose;
    
    //project all points
    projectPoints(mVertices, poseComb.rvec(), poseComb.translation(), cameraMatrix, distCoeffs, vProjPoints);
    
    return vProjPoints;
}

void Object3D::drawVertice(const Point3f &_vertice, Mat &img, const Mat &cameraMatrix, const Mat &distCoeffs, const Affine3d &poseCam) const
{
    //transform using object pose
    vector<Point3f> pobj; pobj.push_back(pose*_vertice);
    Point3f pcam; pcam= poseCam * pose * pobj[0];
    
    //project
    vector<Point2f> vprojVertices;
    projectPoints(pobj, poseCam.rvec(), poseCam.translation(), cameraMatrix, distCoeffs, vprojVertices);
    

    if(pcam.z>0 && 1./pcam.z > 0)
        circle(img, vprojVertices[0], 3./pcam.z, Scalar(255,255,0),2);
}

void Object3D::drawEdge(const ModelEdge &_edge, const Mat &img, const Mat &cameraMatrix, const Mat &distCoeffs, const Affine3d &poseCam) const
{
    vector<Point3f> LineObj; LineObj.push_back(pose*_edge.ptFrom);LineObj.push_back(pose*_edge.ptTo);
    vector<Point3f> LineCam; LineCam.push_back(poseCam*LineObj[0]);LineCam.push_back(poseCam*LineObj[1]);
    
    vector<Point2f> vprojVertices;
    projectPoints(LineObj, poseCam.rvec(), poseCam.translation(), cameraMatrix, distCoeffs, vprojVertices);
    //if(LineCam[0].z>0 && LineCam[1].z>0)
        line(img, vprojVertices[0], vprojVertices[1], Scalar(0,0,255), 2);
    
    
    
}

float rotationVSfrontoparallel(const Vec3d &rvec)
{
    Affine3d cameraRotPose = Affine3d().rotate(rvec);
    Point3f frontoPerp(0.,0,1);
    
    //express frontoPerp in cam
    Point3f frontoPerpCam=cameraRotPose*frontoPerp;
    //get angle with frontoperp from arcsin of length of frontoPerpCam on x,y
    return asin(sqrt(frontoPerpCam.x*frontoPerpCam.x+frontoPerpCam.y*frontoPerpCam.y));
}

bool compareByDiscriminativePower(const DetectionGH &a, const DetectionGH &b)
{
    return a.discriminativePower > b.discriminativePower;
}

bool Object3D::getPose(const IntrinsicCalibration &_mCalib, vector<DetectionGH> mMatches, Affine3d &robotPose, bool init) const
{
    //if don t have 4 measures then problem not solvable
    if(mMatches.size() < 4)
        return false;
    
    //sort matches with respect to discriminative power to ease ransac later
    //put higher discriminative detections first
    std::sort(mMatches.begin(), mMatches.end(), compareByDiscriminativePower);
    
    //create list of 3d points corresponding to detected projections
    vector<Point3f> detectedVertices;
    for(unsigned int i=0;i<mMatches.size();i++)detectedVertices.push_back(mVertices[mMatches[i].id]);
    
    //use opencv function
    //get previous position if there is any
    Vec3d rvec_prec,tvec_prec;
    if(!init)
    {
        rvec_prec=robotPose.rvec();
        tvec_prec=robotPose.translation();
    }
    
    //do a kind of ransac: try different subset to compute pose util find that more than majority agrees
    //if not consider tracker lost
    const unsigned int nbBasePnp=4;//take four points out of set
    unsigned int pointers[nbBasePnp];
    for(unsigned int i=0;i<nbBasePnp;i++)pointers[i]=i;//set first pointers as first elements of list
    
    while(1)
    {
        //want to test all the unsorted subsets of 4 points
        Vec3d rvec,tvec;
        if(!init)    {rvec=rvec_prec;tvec=tvec_prec;}
        
        //create vectors corresponding to subset
        vector<Point3f> subsetVertices;
        vector<Point2f> subsetProjections;
        for(unsigned int i=0;i<nbBasePnp;i++)
        {
            subsetVertices.push_back(detectedVertices[pointers[i]]);
            subsetProjections.push_back(mMatches[pointers[i]].position);
        }
        
        
        //compute pose with subset
        cv::solvePnP(subsetVertices, subsetProjections, _mCalib.cameraMatrix, _mCalib.distCoeffs, rvec, tvec, !init);
        //cv::solvePnPRansac(detectedVertices, pointPositions, cameraMatrix, distCoeffs, rvec, tvec);
        
        //check if estimated transformation is possible (if rotation with respect to image plan > max rotation GH training
        //then would not be able to do this association using GH)
        if(rotationVSfrontoparallel(rvec)<M_PI/4)
        {
            
            //check how many points agree
            float threshold_proj=5.;//set error max to 5 pixels
            unsigned int nbPointAgree=0;
            vector<Point2f> vProjPoints;//project all points
            projectPoints(detectedVertices, rvec, tvec, _mCalib.cameraMatrix, _mCalib.distCoeffs, vProjPoints);
            for(unsigned int i=0;i<mMatches.size();i++)
                if(norm(vProjPoints[i]-mMatches[i].position)<threshold_proj)
                    nbPointAgree++;
            
            //if more than majority agrees then fine, recompute objects pose with all points which agree and return it
            //might be better to instead do a tuckey optimisation with all points, as their might be points rejected with
            //first subset which after some refining would be back in...
            //can also run several iterations of the previous selection and following optimisation
            if(nbPointAgree>mMatches.size()/2)
            {
                vector<Point3f> newSubsetVertices;
                vector<Point2f> newSubsetProjections;
                for(unsigned int i=0;i<mMatches.size();i++)
                    if(norm(vProjPoints[i]-mMatches[i].position)<threshold_proj)
                    {
                        newSubsetVertices.push_back(detectedVertices[i]);
                        newSubsetProjections.push_back(mMatches[i].position);
                    }
                
                cv::solvePnP(newSubsetVertices, newSubsetProjections, _mCalib.cameraMatrix, _mCalib.distCoeffs, rvec, tvec, !init);
                robotPose=Affine3d(rvec,tvec);
                return true;
            }
        }
        
        //if not need to go to next subset:
        //try moving last pointer, if pointer reaches end, then have to move previous pointer
        //recursive if any pointer reaches its end position then moves it previous pointer and put current pointer after it
        //do until first point (pointers[0]) reaches the end (for pointers[0] corresponds to pointPositions.size-nbBasePnp)
        
        //to do that just count how many pointers are stuck at the end:
        int nbPtStuckAtEnd=0;
        for(int i=nbBasePnp-1;i>=0;i--)
            if(pointers[i]==mMatches.size()-nbBasePnp+i)
                nbPtStuckAtEnd++;
        
        if(nbPtStuckAtEnd==0)//if none stuck then can simply move last one
            pointers[nbBasePnp-1]++;
        else if(nbPtStuckAtEnd==nbBasePnp)//all of them are stuck and we didn t find anything good=> lost
            return false;
        else//some of them are at the end
        {
            //need to move the (nbBasePnp-1-nbPtStuckAtEnd)th pointer right and set the others to follow
            pointers[nbBasePnp-1-nbPtStuckAtEnd]++;
            for(unsigned int i=nbBasePnp-1-nbPtStuckAtEnd+1;i<nbBasePnp;i++)
                pointers[i]=pointers[i-1]+1;
        }
        
    }
}


Camera3dModel::Camera3dModel()
{
    vector<Point3f> mVerticesTemp;
    float scale=0.04;//make it about 5cm big
    float w=scale*0.4,h=scale*0.3,z=scale*0.8;
    
    //create vertices from dimensions
    mVerticesTemp.push_back(Point3f(0,0,0));
    mVerticesTemp.push_back(Point3f(w,h,z));
    mVerticesTemp.push_back(Point3f(w,-h,z));
    mVerticesTemp.push_back(Point3f(-w,-h,z));
    mVerticesTemp.push_back(Point3f(-w,h,z));
    
    //from center to image plane
    mEdges.push_back(ModelEdge(mVerticesTemp[0],mVerticesTemp[1]));
    mEdges.push_back(ModelEdge(mVerticesTemp[0],mVerticesTemp[2]));
    mEdges.push_back(ModelEdge(mVerticesTemp[0],mVerticesTemp[3]));
    mEdges.push_back(ModelEdge(mVerticesTemp[0],mVerticesTemp[4]));
    
    //image plane
    mEdges.push_back(ModelEdge(mVerticesTemp[1],mVerticesTemp[2]));
    mEdges.push_back(ModelEdge(mVerticesTemp[2],mVerticesTemp[3]));
    mEdges.push_back(ModelEdge(mVerticesTemp[3],mVerticesTemp[4]));
    mEdges.push_back(ModelEdge(mVerticesTemp[4],mVerticesTemp[1]));
}

    


ThymioBlobModel::ThymioBlobModel()
{
    //vertices in cm, origin is on corner blob
    /*mVertices.push_back(Point3f(0,5.6,3.2));
    mVertices.push_back(Point3f(0,4.8,3.2));
    mVertices.push_back(Point3f(0,0.8,3.2));
    mVertices.push_back(Point3f(0,0,3.2));
    
    mVertices.push_back(Point3f(0.8,4.8,3.2));
    mVertices.push_back(Point3f(0.8,0.8,3.2));
    mVertices.push_back(Point3f(0.8,0,3.2));
    
    mVertices.push_back(Point3f(8.,4.8,3.2));
    mVertices.push_back(Point3f(8.,0.8,3.2));
    mVertices.push_back(Point3f(8.,0,3.2));
    
    mVertices.push_back(Point3f(8.8,5.6,3.2));
    mVertices.push_back(Point3f(8.8,4.8,3.2));
    mVertices.push_back(Point3f(8.8,0.8,3.2));
    mVertices.push_back(Point3f(8.8,0,3.2));*/
    
    //top of the robot
    //model definition:
    // 10             0
    // 11  7       4  1
    //
    // 12  8       5  2
    // 13  9       6  3
    
    //set everything in centimeters and will divide later to convert in meters
    mVertices.push_back(Point3f(8.8,5.6,3.2));
    mVertices.push_back(Point3f(8.8,4.8,3.2));
    mVertices.push_back(Point3f(8.8,0.8,3.2));
    mVertices.push_back(Point3f(8.8,0,3.2));
    
    mVertices.push_back(Point3f(8,4.8,3.2));
    mVertices.push_back(Point3f(8,0.8,3.2));
    mVertices.push_back(Point3f(8,0,3.2));
    
    mVertices.push_back(Point3f(0.8,4.8,3.2));
    mVertices.push_back(Point3f(0.8,0.8,3.2));
    mVertices.push_back(Point3f(0.8,0,3.2));
    
    mVertices.push_back(Point3f(0.,5.6,3.2));
    mVertices.push_back(Point3f(0.,4.8,3.2));
    mVertices.push_back(Point3f(0.,0.8,3.2));
    mVertices.push_back(Point3f(0.,0,3.2));
    
    //define groups in clockwise order
    mGroup3s.push_back(ModelTriplet(0,1,4));
    mGroup3s.push_back(ModelTriplet(10,7,11));
    
    mGroup4s.push_back(ModelQuadruplet(5,2,3,6));
    mGroup4s.push_back(ModelQuadruplet(12,8,9,13));

    //set them in meter and center them
    for(int v=0;v<14;v++)
        mVertices[v]=(mVertices[v]-Point3f(4.4,2.8,0))/100.;


    //back of robot again put in centimeters and origin as left down vertice
    //     16           19
    //  15                  18
    //     14           17
    /*mVertices.push_back(Point3f(0.,0.,0.));
    mVertices.push_back(Point3f(-0.7,0.,0.675));
    mVertices.push_back(Point3f(0.,0.,1.35));
    
    mVertices.push_back(Point3f(8.4,0.,0.));
    mVertices.push_back(Point3f(9.1,0.,0.675));
    mVertices.push_back(Point3f(8.4,0.,1.35));

    for(int v=14;v<20;v++)
        mVertices[v]=(mVertices[v]-Point3f(4.1,2.9,-0.3))/10.;

    mGroup3s.push_back(ModelTriplet(14,15,16));
    mGroup3s.push_back(ModelTriplet(19,18,17));

    //side left
    // 21
    // 
    // 20   22
    mVertices.push_back(Point3f(0.,0.,0.));
    mVertices.push_back(Point3f(0.,0.,0.95));
    mVertices.push_back(Point3f(0.,-0.95,0));
    for(int v=20;v<23;v++)
        mVertices[v]=(mVertices[v]-Point3f(5.5,-5.1,0.45))/10.;
    mGroup3s.push_back(ModelTriplet(20,21,22));
    
    //right left
    //      25
    // 
    // 24   23
    mVertices.push_back(Point3f(0.,0.,0.));
    mVertices.push_back(Point3f(0.,-0.95,0));
    mVertices.push_back(Point3f(0.,0.,0.95));
    for(int v=23;v<26;v++)
        mVertices[v]=(mVertices[v]-Point3f(-5.5,-5.1,0.45))/10.;
    mGroup3s.push_back(ModelTriplet(23,24,25));*/


 
    
    //if want to display edges
    vector<Point3f> mVerticesTop;
    mVerticesTop.push_back(Point3f(-0.65,7.75,3.2));
    mVerticesTop.push_back(Point3f(-0.65,-0.65,3.2));
    mVerticesTop.push_back(Point3f(9.5,-0.65,3.2));
    mVerticesTop.push_back(Point3f(9.5,7.75,3.2));
    mVerticesTop.push_back(Point3f(7.,9.4,3.2));
    mVerticesTop.push_back(Point3f(5.,9.7,3.2));
    mVerticesTop.push_back(Point3f(2.5,9.4,3.2));
    
    //set them in meter and center them
    for(unsigned int v=0;v<mVerticesTop.size();v++)
        mVerticesTop[v]=(mVerticesTop[v]-Point3f(4.4,2.8,0))/100.;
    
    //from center to image plane
    for(unsigned int v=0;v<mVerticesTop.size();v++)
        mEdges.push_back(ModelEdge(mVerticesTop[v],mVerticesTop[(v+1)%mVerticesTop.size()]));
    
    vector<Point3f> mVerticesBottom;
    mVerticesBottom.push_back(Point3f(-0.65,7.75,-1.2));
    mVerticesBottom.push_back(Point3f(-0.65,-0.65,-1.2));
    mVerticesBottom.push_back(Point3f(9.5,-0.65,-1.2));
    mVerticesBottom.push_back(Point3f(9.5,7.75,-1.2));
    mVerticesBottom.push_back(Point3f(7.,9.4,-1.2));
    mVerticesBottom.push_back(Point3f(5.,9.7,-1.2));
    mVerticesBottom.push_back(Point3f(2.5,9.4,-1.2));
    
    //set them in meter and center them
    for(unsigned int v=0;v<mVerticesBottom.size();v++)
        mVerticesBottom[v]=(mVerticesBottom[v]-Point3f(4.4,2.8,0))/100.;
    
    //from center to image plane
    for(unsigned int v=0;v<mVerticesBottom.size();v++)
        mEdges.push_back(ModelEdge(mVerticesBottom[v],mVerticesBottom[(v+1)%mVerticesBottom.size()]));
    
    //vertical edges
    vector<Point3f> mVerticesVertical;
    //create vertices from dimensions
    mVerticesVertical.push_back(Point3f(-0.65,7.75,3.2));
    mVerticesVertical.push_back(Point3f(-0.65,7.75,-1.2));
    mVerticesVertical.push_back(Point3f(-0.65,-0.65,3.2));
    mVerticesVertical.push_back(Point3f(-0.65,-0.65,-1.2));
    mVerticesVertical.push_back(Point3f(9.5,-0.65,3.2));
    mVerticesVertical.push_back(Point3f(9.5,-0.65,-1.2));
    mVerticesVertical.push_back(Point3f(9.5,7.75,3.2));
    mVerticesVertical.push_back(Point3f(9.5,7.75,-1.2));
    //set them in meter and center them
    for(unsigned int v=0;v<mVerticesVertical.size();v++)
        mVerticesVertical[v]=(mVerticesVertical[v]-Point3f(4.4,2.8,0))/100.;
    
    //from center to image plane
    for(unsigned int v=0;v<mVerticesVertical.size()/2;v++)
        mEdges.push_back(ModelEdge(mVerticesVertical[2*v],mVerticesVertical[2*v+1]));


    //define what s going to be used in active search
    //template of top view
    mImage = cv::imread("../data/robot/robotTopCropHalf.png",CV_LOAD_IMAGE_GRAYSCALE);
    //mImage = cv::imread("../data/robot/robotTopCropHalf.png",CV_LOAD_IMAGE_COLOR);

    if(mImage.empty())
    {
        std::cerr << "Could not open " << "../data/robot/robotTopCropHalf.png" << std::endl;
        throw std::runtime_error("Template image from robot file not found!");        
    }

    //position of projection of vertices in top template
    mRobotKeypointPos.push_back(cv::Point2f(100,480));
    mRobotKeypointPos.push_back(cv::Point2f(102,421));
    mRobotKeypointPos.push_back(cv::Point2f(112,136));
    mRobotKeypointPos.push_back(cv::Point2f(114,80));

    mRobotKeypointPos.push_back(cv::Point2f(160,421));
    mRobotKeypointPos.push_back(cv::Point2f(168,137));
    mRobotKeypointPos.push_back(cv::Point2f(170,81));

    mRobotKeypointPos.push_back(cv::Point2f(681,427));
    mRobotKeypointPos.push_back(cv::Point2f(679,141));
    mRobotKeypointPos.push_back(cv::Point2f(679,83));

    mRobotKeypointPos.push_back(cv::Point2f(742,487));
    mRobotKeypointPos.push_back(cv::Point2f(740,428));
    mRobotKeypointPos.push_back(cv::Point2f(736,141));
    mRobotKeypointPos.push_back(cv::Point2f(736,85));

    //some more keypoints with a bit of texture
    mRobotKeypointPos.push_back(cv::Point2f(422,86));

    //next points are defined on bumps which appearance depends highly on illumination
    //Cross correlation is not robut enough to do direct image similarity measure
    //if need more points in future might have to use orientation of gradients
    /*mRobotKeypointPos.push_back(cv::Point2f(420,456));
    mRobotKeypointPos.push_back(cv::Point2f(525,570));
    mRobotKeypointPos.push_back(cv::Point2f(420,684));
    mRobotKeypointPos.push_back(cv::Point2f(310,566));

    mRobotKeypointPos.push_back(cv::Point2f(422,227));*/

    //use half resolution
    for(unsigned int i=0;i<mRobotKeypointPos.size();i++)
        mRobotKeypointPos[i] = mRobotKeypointPos[i]/2.;


    for(int i=0;i<14;i++)
        mVerticesTopPos.push_back(mRobotKeypointPos[i]);

    
}

}
