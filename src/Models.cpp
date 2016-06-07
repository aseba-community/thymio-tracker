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

Vec3d toVec(Point3f _p)
{
    return Vec3d(_p.x,_p.y,_p.z);
}
Point3f toPoint(Vec3d _p)
{
    return Point3f(_p[0],_p[1],_p[2]);
}

void Object3D::draw(Mat& img, const Mat& cameraMatrix, const Mat& distCoeffs, const Affine3d& poseCam) const
{
    //for(unsigned int e=0;e<mEdges.size();e++)
    //    drawEdge(mEdges[e], img, cameraMatrix, distCoeffs, poseCam);
    
    for(unsigned int e=0;e<mEdgesTrack.size();e++)
        drawEdgeTrack(mEdgesTrack[e], img, cameraMatrix, distCoeffs, poseCam);
    
    for(unsigned int v=0;v<mVertices.size();v++)
        drawVertice(mVertices[v], img, cameraMatrix, distCoeffs, poseCam);

    for(unsigned int v=0;v<mPlanarSurfaces.size();v++)
        drawSurface(mPlanarSurfaces[v], img, cameraMatrix, distCoeffs, poseCam);
    
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
void Object3D::drawSurface(const planarSurface &_edge, const Mat &img, const Mat &cameraMatrix, const Mat &distCoeffs, const Affine3d &poseCam) const
{
    vector<Point3f> ObjPoints;
    ObjPoints.push_back(_edge.center);
    ObjPoints.push_back(_edge.center+toPoint(0.01*_edge.normal));
    ObjPoints.push_back(_edge.center+toPoint(_edge.radius1*_edge.b1+_edge.radius2*_edge.b2));
    ObjPoints.push_back(_edge.center+toPoint(_edge.radius1*_edge.b1-_edge.radius2*_edge.b2));
    ObjPoints.push_back(_edge.center+toPoint(_edge.radius1*-_edge.b1-_edge.radius2*_edge.b2));
    ObjPoints.push_back(_edge.center+toPoint(_edge.radius1*-_edge.b1+_edge.radius2*_edge.b2));

    vector<Point3f> LineObj;
    for(int i=0;i<ObjPoints.size();i++) 
        LineObj.push_back(pose*ObjPoints[i]);

    vector<Point3f> LineCam;
    for(int i=0;i<LineObj.size();i++) 
        LineCam.push_back(poseCam*LineObj[i]);
    
    vector<Point2f> vprojVertices;
    projectPoints(LineObj, poseCam.rvec(), poseCam.translation(), cameraMatrix, distCoeffs, vprojVertices);

    Point3f ptLine = poseCam * pose * _edge.center;
    Vec3d ray = Vec3d(ptLine.x,ptLine.y,ptLine.z);
    ray = ray / norm(ray);
    Vec3d normal_cam = poseCam.rotation() * pose.rotation() * _edge.normal;
    float view = -normal_cam.dot(ray);

    //if(LineCam[0].z>0 && LineCam[0].z - LineCam[1].z >0)
    if(view > 0)
    {
        //std::cout<<"surf viewScore"<<view<<std::endl;
        //print normal
        line(img, vprojVertices[0], vprojVertices[1], Scalar(0,255,255), 2); 

        //print surface
        for(int i=0;i<4;i++)
            line(img, vprojVertices[2+i], vprojVertices[2+((i+1) % 4)], Scalar(255,0,255), 2);   
    }
       

}


void Object3D::drawEdgeTrack(const ModelEdgeTrack &_edge, const Mat &img, const Mat &cameraMatrix, const Mat &distCoeffs, const Affine3d &poseCam) const
{
    //check visibility
    Vec3d normal1_cam = poseCam.rotation() * pose.rotation() * _edge.normal1;
    Vec3d normal2_cam = poseCam.rotation() * pose.rotation() * _edge.normal2;

    //get angle between normals and ray
    Point3f ptLine = poseCam * pose * _edge.ptFrom;
    Vec3d ray = Vec3d(ptLine.x,ptLine.y,ptLine.z);
    ray = ray / norm(ray);
    float angleView1 = acos(normal1_cam.dot(ray));
    float angleView2 = acos(normal2_cam.dot(ray));

    vector<Point3f> LineObj; LineObj.push_back(pose*_edge.ptFrom);LineObj.push_back(pose*_edge.ptTo);
    
    vector<Point2f> vprojVertices;
    projectPoints(LineObj, poseCam.rvec(), poseCam.translation(), cameraMatrix, distCoeffs, vprojVertices);

    //edge is part of silouette if one face is visible and the other not.
    if( (angleView1 > M_PI/2 && angleView2 < M_PI/2) || (angleView1 < M_PI/2 && angleView2 > M_PI/2))
        line(img, vprojVertices[0], vprojVertices[1], Scalar(0,0,255), 2);   


    //print normals
    vector<Point3f> LineNormals;
    LineNormals.push_back((_edge.ptFrom + _edge.ptTo)/2.);
    LineNormals.push_back((_edge.ptFrom + _edge.ptTo)/2. + toPoint(0.01*_edge.normal1));
    LineNormals.push_back((_edge.ptFrom + _edge.ptTo)/2. + toPoint(0.01*_edge.normal2));

    vector<Point3f> LineNormals_p;
    for(int i=0;i<3;i++)
        LineNormals_p.push_back(pose * LineNormals[i]);

    vector<Point2f> vprojNormals;
    projectPoints(LineNormals_p, poseCam.rvec(), poseCam.translation(), cameraMatrix, distCoeffs, vprojNormals);
    line(img, vprojNormals[0], vprojNormals[1], Scalar(0,255,0), 1);  
    line(img, vprojNormals[0], vprojNormals[2], Scalar(255,0,0), 1);  
    
}

float rotationVSfrontoparallel(const Vec3d &rvec)
{
    Affine3d cameraRotPose = Affine3d().rotate(rvec);
    Point3f frontoPerp(0.,0,-1);
    
    //express frontoPerp in cam
    Point3f frontoPerpCam=cameraRotPose*frontoPerp;
    //get angle with frontoperp
    return acos(frontoPerpCam.z);
}

bool compareByDiscriminativePower(const DetectionGH &a, const DetectionGH &b)
{
    return a.discriminativePower > b.discriminativePower;
}

//get all subsets of p elements out of a n elements set
bool getNextSetPointers(unsigned int *pointers, int n, int p)
{
    int nbPtStuckAtEnd=0;
    for(int i=p-1;i>=0;i--)
        if(pointers[i]==n-p+i)
            nbPtStuckAtEnd++;
    
    if(nbPtStuckAtEnd==0)//if none stuck then can simply move last one
        pointers[p-1]++;
    else if(nbPtStuckAtEnd==p)//all of them are stuck and we didn t find anything good=> lost
    {
        //std::cout<< "ratio_max = "<<ratio_max<<std::endl;
        return false;
    }
    else//some of them are at the end
    {
        //need to move the (nbBasePnp-1-nbPtStuckAtEnd)th pointer right and set the others to follow
        pointers[p-1-nbPtStuckAtEnd]++;
        for(unsigned int i=p-1-nbPtStuckAtEnd+1;i<p;i++)
            pointers[i]=pointers[i-1]+1;
    }
    return true;
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
    if(init)
    {
        rvec_prec=robotPose.rvec();
        tvec_prec=robotPose.translation();
    }
    else
    {
        rvec_prec = Vec3d(0.,M_PI,0.);
        tvec_prec= Vec3d(0.,0.,0.4);
    }
    
    //do a kind of ransac: try different subset to compute pose util find that more than majority agrees
    //if not consider tracker lost
    const unsigned int nbBasePnp=4;//take four points out of set
    unsigned int pointers[nbBasePnp];
    for(unsigned int i=0;i<nbBasePnp;i++)pointers[i]=i;//set first pointers as first elements of list

    //for testing: ratio max agreeing matches
    float ratio_max = 0;
   
    while(1)
    {
        //want to test all the unsorted subsets of 4 points
        Vec3d rvec,tvec;
        //if(!init)    {rvec=rvec_prec;tvec=tvec_prec;}
        rvec=rvec_prec;tvec=tvec_prec;
        
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
        //std::cout<< "rotationVSfrontoparallel(rvec) = "<<rotationVSfrontoparallel(rvec)<<std::endl;
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
                //std::cout<< "ratio = "<<(float)nbPointAgree/mMatches.size()<<std::endl;
                //std::cout<< "rotationVSfrontoparallel(rvec) = "<<rotationVSfrontoparallel(rvec)<<std::endl;
                return true;
            }

            if((float)nbPointAgree/mMatches.size()>ratio_max)
                ratio_max = (float)nbPointAgree/mMatches.size();
        }
        
        //if not need to go to next subset:
        //try moving last pointer, if pointer reaches end, then have to move previous pointer
        //recursive if any pointer reaches its end position then moves it previous pointer and put current pointer after it
        //do until first point (pointers[0]) reaches the end (for pointers[0] corresponds to pointPositions.size-nbBasePnp)
        
        //to do that just count how many pointers are stuck at the end:
        bool nextSetAvailable = getNextSetPointers(&pointers[0],mMatches.size(),nbBasePnp);

        if(!nextSetAvailable)
            return false;

        /*int nbPtStuckAtEnd=0;
        for(int i=nbBasePnp-1;i>=0;i--)
            if(pointers[i]==mMatches.size()-nbBasePnp+i)
                nbPtStuckAtEnd++;
        
        if(nbPtStuckAtEnd==0)//if none stuck then can simply move last one
            pointers[nbBasePnp-1]++;
        else if(nbPtStuckAtEnd==nbBasePnp)//all of them are stuck and we didn t find anything good=> lost
        {
            //std::cout<< "ratio_max = "<<ratio_max<<std::endl;
            return false;
        }
        else//some of them are at the end
        {
            //need to move the (nbBasePnp-1-nbPtStuckAtEnd)th pointer right and set the others to follow
            pointers[nbBasePnp-1-nbPtStuckAtEnd]++;
            for(unsigned int i=nbBasePnp-1-nbPtStuckAtEnd+1;i<nbBasePnp;i++)
                pointers[i]=pointers[i-1]+1;
        }*/
        
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

void ThymioBlobModel::setBlobModel()
{
    //top of the robot
    //model definition:
    // 10             0
    // 11  7       4  1
    //
    // 12  8       5  2
    // 13  9       6  3
    
    //set everything in centimeters and will divide later to convert in meters
    mVertices.push_back(Point3f(0.044,0.036,0.0305));
    mVertices.push_back(Point3f(0.044,0.028,0.0305));
    mVertices.push_back(Point3f(0.044,-0.012,0.0305));
    mVertices.push_back(Point3f(0.044,-0.02,0.0305));

    mVertices.push_back(Point3f(0.036,0.028,0.0305));
    mVertices.push_back(Point3f(0.036,-0.012,0.0305));
    mVertices.push_back(Point3f(0.036,-0.02,0.0305));

    mVertices.push_back(Point3f(-0.036,0.028,0.0305));
    mVertices.push_back(Point3f(-0.036,-0.012,0.0305));
    mVertices.push_back(Point3f(-0.036,-0.02,0.0305));

    mVertices.push_back(Point3f(-0.044,0.036,0.0305));
    mVertices.push_back(Point3f(-0.044,0.028,0.0305));
    mVertices.push_back(Point3f(-0.044,-0.012,0.0305));
    mVertices.push_back(Point3f(-0.044,-0.02,0.0305));
    
    //define groups in clockwise order
    mGroup3s.push_back(ModelTriplet(0,1,4));
    mGroup3s.push_back(ModelTriplet(10,7,11));
    
    mGroup4s.push_back(ModelQuadruplet(5,2,3,6));
    mGroup4s.push_back(ModelQuadruplet(12,8,9,13));
}
void ThymioBlobModel::setEdgePlotModel()
{
    //if want to display edges
    vector<Point3f> mVerticesTop;
    mVerticesTop.push_back(Point3f(-0.0505,0.0575,0.0305));
    mVerticesTop.push_back(Point3f(-0.0505,-0.0265,0.0305));
    mVerticesTop.push_back(Point3f(0.051,-0.0265,0.0305));
    mVerticesTop.push_back(Point3f(0.051,0.0575,0.0305));
    mVerticesTop.push_back(Point3f(0.026,0.074,0.0305));
    mVerticesTop.push_back(Point3f(0.006,0.077,0.0305));
    mVerticesTop.push_back(Point3f(-0.019,0.074,0.0305));
    
    //from center to image plane
    for(unsigned int v=0;v<mVerticesTop.size();v++)
        mEdges.push_back(ModelEdge(mVerticesTop[v],mVerticesTop[(v+1)%mVerticesTop.size()]));
    
    vector<Point3f> mVerticesBottom;
    mVerticesBottom.push_back(Point3f(-0.0505,0.0575,-0.0135));
    mVerticesBottom.push_back(Point3f(-0.0505,-0.0265,-0.0135));
    mVerticesBottom.push_back(Point3f(0.051,-0.0265,-0.0135));
    mVerticesBottom.push_back(Point3f(0.051,0.0575,-0.0135));
    mVerticesBottom.push_back(Point3f(0.026,0.074,-0.0135));
    mVerticesBottom.push_back(Point3f(0.006,0.077,-0.0135));
    mVerticesBottom.push_back(Point3f(-0.019,0.074,-0.0135));
  
    //from center to image plane
    for(unsigned int v=0;v<mVerticesBottom.size();v++)
        mEdges.push_back(ModelEdge(mVerticesBottom[v],mVerticesBottom[(v+1)%mVerticesBottom.size()]));
    
    //vertical edges
    vector<Point3f> mVerticesVertical;
    //create vertices from dimensions
    mVerticesVertical.push_back(Point3f(-0.0505,0.0575,0.0305));
    mVerticesVertical.push_back(Point3f(-0.0505,0.0575,-0.0135));
    mVerticesVertical.push_back(Point3f(-0.0505,-0.0265,0.0305));
    mVerticesVertical.push_back(Point3f(-0.0505,-0.0265,-0.0135));
    mVerticesVertical.push_back(Point3f(0.051,-0.0265,0.0305));
    mVerticesVertical.push_back(Point3f(0.051,-0.0265,-0.0135));
    mVerticesVertical.push_back(Point3f(0.051,0.0575,0.0305));
    mVerticesVertical.push_back(Point3f(0.051,0.0575,-0.0135));
    
    //from center to image plane
    for(unsigned int v=0;v<mVerticesVertical.size()/2;v++)
        mEdges.push_back(ModelEdge(mVerticesVertical[2*v],mVerticesVertical[2*v+1]));
}
void ThymioBlobModel::setEdgeTrackModel()
{
    Point3f heightRobot(0.,0.,0.0445);

    //define edges for tracking
    vector<Point3f> mVerticesTrackTop;
    mVerticesTrackTop.push_back(Point3f(-0.055,0.055,0.0305));
    mVerticesTrackTop.push_back(Point3f(-0.055,-0.0295,0.0305));
    mVerticesTrackTop.push_back(Point3f(0.055,-0.0295,0.0305));
    mVerticesTrackTop.push_back(Point3f(0.055,0.055,0.0305));

    float sqrt2_2 = sqrt(2.)/2.;
    //bissectrice of edge
    Vec3d mNormals1TrackTop(0.,0.,-1.);
    
    vector<Vec3d> mNormals2TrackTop;
    mNormals2TrackTop.push_back(Vec3d(sqrt2_2,0,0));
    mNormals2TrackTop.push_back(Vec3d(0.,sqrt2_2,0));
    mNormals2TrackTop.push_back(Vec3d(-sqrt2_2,0,0));

    //radius of rounding of edge
    /*vector<float> mEdgesRounding;
    mEdgesRounding.push_back(0.004);
    mEdgesRounding.push_back(0.0015);
    mEdgesRounding.push_back(0.004);*/

    for(unsigned int v=0;v<mVerticesTrackTop.size()-1;v++)
        //mEdgesTrack.push_back(ModelEdgeTrack(mVerticesTrackTop[v],mVerticesTrackTop[(v+1)],mNormalsTrackTop[v],mEdgesRounding[v]));
        mEdgesTrack.push_back(ModelEdgeTrack(mVerticesTrackTop[v],mVerticesTrackTop[(v+1)],mNormals1TrackTop,mNormals2TrackTop[v]));


    vector<Point3f> mVerticesTrackTopRound;
    vector<Vec3d> mNormals2TrackTopRound;

    //round bit
    int nbRoundCut = 4;
    //more elliptic than perfect circle => define two radii
    float radiusFront = 0.08;
    float radiusSides = 0.078;
    for(int i=-nbRoundCut;i<=nbRoundCut;i++)
    {
        float angle = - i * M_PI / (4. * nbRoundCut);
        float radius = radiusSides * abs(i) / nbRoundCut + radiusFront * (nbRoundCut - abs(i)) / nbRoundCut;
        mVerticesTrackTopRound.push_back(Point3f(radius * sin(angle),radius * cos(angle),0.0305));    
        float angleSeg = - ((float)i+0.5) * M_PI / (4. * nbRoundCut);
        mNormals2TrackTopRound.push_back(Vec3d(- sin(angleSeg),- cos(angleSeg),0.));    
    }

    for(unsigned int v=0;v<mVerticesTrackTopRound.size()-1;v++)
        mEdgesTrack.push_back(ModelEdgeTrack(mVerticesTrackTopRound[v],mVerticesTrackTopRound[(v+1)],mNormals1TrackTop,mNormals2TrackTopRound[v]));

    //edges from bottom
    Vec3d mNormals1TrackBottom(0.,0.,1.);
    int nbEdgesTop = mEdgesTrack.size();
    for(int i=0; i< nbEdgesTop;i++)
        mEdgesTrack.push_back(ModelEdgeTrack(mEdgesTrack[i].ptFrom - heightRobot,mEdgesTrack[i].ptTo - heightRobot,mNormals1TrackBottom,mEdgesTrack[i].normal2));
    

    //vertical edges
    mEdgesTrack.push_back(ModelEdgeTrack(mVerticesTrackTop[0],mVerticesTrackTop[0] - heightRobot,Vec3d(sqrt2_2,-sqrt2_2,0.),Vec3d(1.,0.,0.)) );
    mEdgesTrack.push_back(ModelEdgeTrack(mVerticesTrackTop[1],mVerticesTrackTop[1] - heightRobot,Vec3d(0.,1.,0.),Vec3d(1.,0.,0.)) );
    mEdgesTrack.push_back(ModelEdgeTrack(mVerticesTrackTop[2],mVerticesTrackTop[2] - heightRobot,Vec3d(0.,1.,0.),Vec3d(-1.,0.,0.)) );
    mEdgesTrack.push_back(ModelEdgeTrack(mVerticesTrackTop[3],mVerticesTrackTop[3] - heightRobot,Vec3d(-sqrt2_2,-sqrt2_2,0.),Vec3d(-1.,0.,0.)) );



}
    


ThymioBlobModel::ThymioBlobModel()
{
    setBlobModel();
    setEdgePlotModel();
    setEdgeTrackModel();
    setSurfacesModel();
    readSurfaceLearned();
}

void ThymioBlobModel::setSurfacesModel()
{
    //remark: order matters as we use simetry afeterward... might be better to change it to a model for one side only

    //wheels
    //mPlanarSurfaces.push_back(planarSurface(cv::Point3f(0.055,0,0), cv::Vec3d(0.,1.,0.), cv::Vec3d(0.,0.,1.), 0.03));
    //mPlanarSurfaces.push_back(planarSurface(cv::Point3f(-0.055,0,0), cv::Vec3d(0.,-1.,0.), cv::Vec3d(0.,0.,1.), 0.03));
    mPlanarSurfaces.push_back(planarSurface(cv::Point3f(0.055, 0.015,0.005), cv::Vec3d(0.,1.,0.), cv::Vec3d(0.,0.,1.), 0.01,0.015));
    mPlanarSurfaces.push_back(planarSurface(cv::Point3f(0.055,-0.015,0.005), cv::Vec3d(0.,1.,0.), cv::Vec3d(0.,0.,1.), 0.01,0.015));
    //top blobs
    mPlanarSurfaces.push_back(planarSurface(cv::Point3f( 0.04, -0.015,0.031), cv::Vec3d(1.,0.,0.), cv::Vec3d(0.,1.,0.), 0.01));
    mPlanarSurfaces.push_back(planarSurface(cv::Point3f( 0.04,  0.033,0.031), cv::Vec3d(1.,0.,0.), cv::Vec3d(0.,1.,0.), 0.01));
    //back rectangle
    mPlanarSurfaces.push_back(planarSurface(cv::Point3f( 0.03, -0.0295,0.012), cv::Vec3d(1.,0.,0.), cv::Vec3d(0.,0.,1.), 0.01));
    
    //front rectanglular bits
    int nbRoundCut = 2;
    //more elliptic than perfect circle => define two radii
    float radiusFront = 0.08;
    float radiusSides = 0.078;
    for(int i=-nbRoundCut;i<=nbRoundCut;i++)
    {
        float angle = - i * M_PI / (4.7 * nbRoundCut);
        float radius = radiusSides * abs(i) / nbRoundCut + radiusFront * (nbRoundCut - abs(i)) / nbRoundCut;
        Point3f center(radius * sin(angle),radius * cos(angle),0.013);    
        mPlanarSurfaces.push_back(planarSurface(center,cv::Vec3d(-cos(angle),sin(angle),0.),cv::Vec3d(0,0,1.),0.01,0.005));    
    }


    mPlanarSurfaces.push_back(planarSurface(cv::Point3f( -0.03, -0.0295,0.012), cv::Vec3d(1.,0.,0.), cv::Vec3d(0.,0.,1.), 0.01));
    mPlanarSurfaces.push_back(planarSurface(cv::Point3f(-0.04,  0.033,0.031), cv::Vec3d(1.,0.,0.), cv::Vec3d(0.,1.,0.), 0.01));
    mPlanarSurfaces.push_back(planarSurface(cv::Point3f(-0.04, -0.015,0.031), cv::Vec3d(1.,0.,0.), cv::Vec3d(0.,1.,0.), 0.01));
    mPlanarSurfaces.push_back(planarSurface(cv::Point3f(-0.055,-0.015,0.005), cv::Vec3d(0.,-1.,0.), cv::Vec3d(0.,0.,1.), 0.01,0.015));
    mPlanarSurfaces.push_back(planarSurface(cv::Point3f(-0.055, 0.015,0.005), cv::Vec3d(0.,-1.,0.), cv::Vec3d(0.,0.,1.), 0.01,0.015)); 

}
    
void ThymioBlobModel::loadTrackingModel(cv::FileStorage& robotModelStorage)
{
    //define what s going to be used in active search
    //template of top view
    //mImage = cv::imread("../data/robot/robotTopCropHalf.png",CV_LOAD_IMAGE_GRAYSCALE);
    //mImage = cv::imread("../data/robot/robotTopCropHalf.png",CV_LOAD_IMAGE_COLOR);

    //robotModelStorage.release();
    //cv::FileStorage fsw("../data/robot/robotTrackInfo.xml", cv::FileStorage::WRITE);
    //cv::write(fsw, "image", mImage);
    //fsw.release();

    cv::read(robotModelStorage["image"], mImage);
    robotModelStorage.release();



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

void Object3D::allocateSurfaceLearning()
{
    for(unsigned int v=0;v<mPlanarSurfaces.size();v++)
        mPlanarSurfaces[v].allocateLearning();

}
void Object3D::learnAppearance(cv::Mat &img, const IntrinsicCalibration &_mCalib, const cv::Affine3d& poseCam)
{
    for(unsigned int v=0;v<mPlanarSurfaces.size();v++)
    {
        planarSurface &surf = mPlanarSurfaces[v];

        //project surface corners to check if visible and get homography from image to surface model
        Point3f ptLine = poseCam * pose * surf.center;
        Vec3d ray = Vec3d(ptLine.x,ptLine.y,ptLine.z);        ray = ray / norm(ray);
        Vec3d normal_cam = poseCam.rotation() * pose.rotation() * surf.normal;
        float viewScore = -normal_cam.dot(ray);

        if(viewScore > 0)
        {
            vector<Point3f> ObjPoints;
            ObjPoints.push_back(surf.center+toPoint(surf.radius1*surf.b1+surf.radius2*surf.b2));
            ObjPoints.push_back(surf.center+toPoint(surf.radius1*surf.b1-surf.radius2*surf.b2));
            ObjPoints.push_back(surf.center+toPoint(surf.radius1*-surf.b1-surf.radius2*surf.b2));
            ObjPoints.push_back(surf.center+toPoint(surf.radius1*-surf.b1+surf.radius2*surf.b2));

            vector<Point3f> LineObj;
            for(int i=0;i<ObjPoints.size();i++) 
                LineObj.push_back(pose*ObjPoints[i]);
            
            vector<Point2f> vprojVertices;
            projectPoints(LineObj, poseCam.rvec(), poseCam.translation(), _mCalib.cameraMatrix, _mCalib.distCoeffs, vprojVertices);

            //get homography
            vector<Point2f> modelPoints;
            modelPoints.push_back(Point2f(surf.mImage.size().width,0));
            modelPoints.push_back(Point2f(surf.mImage.size().width,surf.mImage.size().height));
            modelPoints.push_back(Point2f(0,surf.mImage.size().height));
            modelPoints.push_back(Point2f(0,0));

            cv::Mat homography = cv::findHomography(vprojVertices, modelPoints);

            //warp current image in it
            cv::Mat patchCurr( surf.mImage.size().height, surf.mImage.size().width, img.type() );
            cv::warpPerspective( img, patchCurr, homography, patchCurr.size() );

            //accumulate
            for(int i=0;i<surf.mImage.size().height;i++)
                for(int j=0;j<surf.mImage.size().width;j++)
                    surf.mImage.at<float>(i,j) += viewScore* (float)patchCurr.at<unsigned char>(i,j);

            surf.weight += viewScore;

            //debug
            /*if(surf.weight > 0.1)
            {
                char fileName[200];
                sprintf(fileName, "output/surface%d.png", v);

                //convert to uchar
                for(int i=0;i<surf.mImage.size().height;i++)
                    for(int j=0;j<surf.mImage.size().width;j++)
                        patchCurr.at<unsigned char>(i,j) = surf.mImage.at<float>(i,j)/surf.weight;

                //save
                cv::imwrite(fileName,patchCurr);

            }*/



        }

    }

}

void Object3D::writeSurfaceLearned()
{
    //surfaces should be symetric (6 first with 6 last and 6th should be symetric with itself)
    std::vector<cv::Mat> flippedImages;flippedImages.resize(mPlanarSurfaces.size());
    std::vector<float> weightBuff;weightBuff.resize(mPlanarSurfaces.size());

    for(unsigned int v=0;v<mPlanarSurfaces.size();v++)
    {
        planarSurface &surf = mPlanarSurfaces[v];
        cv::flip(surf.mImage,flippedImages[v],1);
        weightBuff[v] = surf.weight;
    }

    for(unsigned int v=0;v<mPlanarSurfaces.size();v++)
    {
        planarSurface &surf = mPlanarSurfaces[v];

        surf.weight += weightBuff[mPlanarSurfaces.size()-1-v];

        for(int i=0;i<surf.mImage.size().height;i++)
            for(int j=0;j<surf.mImage.size().width;j++)
                surf.mImage.at<float>(i,j) += flippedImages[mPlanarSurfaces.size()-1-v].at<float>(i,j);        
    }


    //convert mImage which is in float to unsigned char as it will be used during tracking
    for(unsigned int v=0;v<mPlanarSurfaces.size();v++)
    {
        planarSurface &surf = mPlanarSurfaces[v];
        cv::Mat buff( surf.mImage.size().height, surf.mImage.size().width, CV_8UC1);
        for(int i=0;i<surf.mImage.size().height;i++)
            for(int j=0;j<surf.mImage.size().width;j++)
                buff.at<unsigned char>(i,j) = surf.mImage.at<float>(i,j)/surf.weight;

        surf.mImage.release();
        buff.copyTo(surf.mImage);
    }

    //save to xml file
    std::vector<cv::Mat> imageVector;
    for(unsigned int v=0;v<mPlanarSurfaces.size();v++)
        imageVector.push_back(mPlanarSurfaces[v].mImage);


    cv::FileStorage store("modelSurfaces.xml.gz", cv::FileStorage::WRITE);
    cv::write(store,"imageVector",imageVector);
    store.release();

    //read debug
    /*std::vector<cv::Mat> imageVectorr;
    cv::FileStorage storer("modelSurfaces.xml.gz", cv::FileStorage::READ);
    cv::FileNode n1 = storer["imageVector"];
    cv::read(n1,imageVectorr);
    storer.release();

    for(unsigned int v=0;v<mPlanarSurfaces.size();v++)
    {
        char fileName[200];
        sprintf(fileName, "output/surface%d.png", v);
        cv::imwrite(fileName,imageVectorr[v]);
    }*/

}

void Object3D::readSurfaceLearned()
{
    std::vector<cv::Mat> imageVectorr;
    cv::FileStorage storer("modelSurfaces.xml.gz", cv::FileStorage::READ);
    cv::FileNode n1 = storer["imageVector"];
    cv::read(n1,imageVectorr);
    storer.release();

    for(unsigned int v=0;v<mPlanarSurfaces.size();v++)
    {
        //cv::Mat detected_edges;
        //cv::blur( imageVectorr[v], detected_edges, Size(3,3) );
        //cv::Canny( detected_edges, imageVectorr[v], 50., 50.*3., 3 );

        //equalizeHist( imageVectorr[v], imageVectorr[v] );

        imageVectorr[v].copyTo(mPlanarSurfaces[v].mImage);

        //ouput as files to check them out
        char fileName[200];
        sprintf(fileName, "output/surface%d.png", v);
        cv::imwrite(fileName,imageVectorr[v]);
    }
    
}

//#define USE_SIMIL_MAX // for NCC use it, for SQ search for min => don t defien it

//info from NCC search to be used by PnP
struct NCCsearchInfo {
    cv::Point3f center;//position of the center of surface in model
    float viewScore;//the more frontoparallel, the more it goes towards 1, goes to 0 otherwise

    cv::Mat NCCpatchScore;//NCC score of surface reprojection in current image
    cv::Point patchCoordinates;//coordinate of the NCCpatchScore (we are not searching the whole image)
    cv::Point relMaxLoc;

    //retrieve NCC score if picking center of surface at _p, weight NCC score with viewScore
    float getNCCscore(cv::Point _p)
    {
        //check if _p is in patch
        Point p = _p-patchCoordinates;
        if(p.x>=0 && p.y>=0 && p.x<NCCpatchScore.size().width && p.y<NCCpatchScore.size().height)
            return viewScore * NCCpatchScore.at<float>(p);
        else
#ifdef USE_SIMIL_MAX
            return 0;
#else
            return 1e2;
#endif
    }

    cv::Point getMaxLoc(){
        return relMaxLoc+patchCoordinates;
    }
        
} ;

/*void Object3D::track(const cv::Mat &img, const IntrinsicCalibration &_mCalib, const cv::Affine3d& prevPoseCam, cv::Affine3d& poseCam) const
{
    //TODO: multiscale approach ? can simply scale image with respect to object distance
    //TODO: try template matching with other similarity functions: NCC has wide basin, more narrow optimum would be better for RANSAC

    //project textured planar surfaces to current image using the previous pose
    //and do NCC search to have 2D correspondance for all of them and then PnP
    //to retrieve the 3D pose from the sets of 2D matches

    
    

    //cv::Mat detected_edges;
    //cv::blur( img, detected_edges, Size(3,3) );
    //cv::Canny( detected_edges, img, 50., 50.*3., 3 );


    cv::Mat MDebugImg;
    img.copyTo(MDebugImg);
    //cv::Mat MDebugImgMask;
    //img.copyTo(MDebugImgMask);    
    cv::Mat MDebugImgWarp;
    img.copyTo(MDebugImgWarp);
    cv::Mat MDebugImgScore;
    img.copyTo(MDebugImgScore);

    int window_search_size = 32;
    int half_window_size = window_search_size/2;

    //information to keep for PnP
    std::vector<NCCsearchInfo> mSearchInfo;

    std::cout<<"compute NCC scores"<<std::endl;

    for(unsigned int v=0;v<mPlanarSurfaces.size();v++)
    {
        //std::cout<<"loop surf["<<v<<"]"<<std::endl;
        const planarSurface &surf = mPlanarSurfaces[v];

        //project surface corners to check if visible and get homography from image to surface model
        Point3f ptLine = prevPoseCam * pose * surf.center;
        Vec3d ray = Vec3d(ptLine.x,ptLine.y,ptLine.z);        ray = ray / norm(ray);
        Vec3d normal_cam = prevPoseCam.rotation() * pose.rotation() * surf.normal;
        float viewScore = -normal_cam.dot(ray);

        if(viewScore > 0.1)
        {
            //std::cout<<"surf viewScore "<<viewScore<<std::endl;
            mSearchInfo.push_back(NCCsearchInfo());
            NCCsearchInfo &mNewINFO = mSearchInfo[mSearchInfo.size()-1];
            mNewINFO.center = surf.center;
            mNewINFO.viewScore = viewScore;

            vector<Point3f> ObjPoints;
            ObjPoints.push_back(surf.center+toPoint(surf.radius1*surf.b1+surf.radius2*surf.b2));
            ObjPoints.push_back(surf.center+toPoint(surf.radius1*surf.b1-surf.radius2*surf.b2));
            ObjPoints.push_back(surf.center+toPoint(-surf.radius1*surf.b1-surf.radius2*surf.b2));
            ObjPoints.push_back(surf.center+toPoint(-surf.radius1*surf.b1+surf.radius2*surf.b2));

            vector<Point3f> LineObj;
            for(int i=0;i<ObjPoints.size();i++) 
                LineObj.push_back(pose*ObjPoints[i]);
            
            //get position of surface corners in previous image
            vector<Point2f> vprojVertices;
            projectPoints(LineObj, prevPoseCam.rvec(), prevPoseCam.translation(), _mCalib.cameraMatrix, _mCalib.distCoeffs, vprojVertices);
            

            //for(int i=0;i<vprojVertices.size();i++) 
            //    std::cout<<"vprojVertices = "<<vprojVertices[i]<<std::endl;

            //get the minimal support rectangle which contains the projected surface
            Rect box = cv::boundingRect(cv::Mat(vprojVertices));
            //std::cout<<"box = "<<box<<std::endl;

            //get projected corners coordinates in the bounding box
            //Point vprojVertices_bb[4];
            //for(int v=0;v<4;v++)
            //    vprojVertices_bb[v] = (vprojVertices[v]-Point2f(box.x, box.y));
            vector<Point> vprojVertices_bb;
            for(int v=0;v<vprojVertices.size();v++)
            {
                vprojVertices_bb.push_back(vprojVertices[v]-Point2f(box.x, box.y));
                //std::cout<<vprojVertices[v]<<std::endl;
            }

            //create the corresponding mask in box wheer surface is defined
            cv::Mat mask = cv::Mat::zeros( box.size(), CV_8UC1 );
            cv::fillConvexPoly(mask, &vprojVertices_bb[0], 4, 1);

            //DEBUG: plot those on the image to check everything is fine
            //mask.copyTo(MDebugImgMask(box));

            //get homography from surface model to current image ROI
            vector<Point2f> modelPoints;
            modelPoints.push_back(Point2f(surf.mImage.size().width,0));
            modelPoints.push_back(Point2f(surf.mImage.size().width,surf.mImage.size().height));
            modelPoints.push_back(Point2f(0,surf.mImage.size().height));
            modelPoints.push_back(Point2f(0,0));

            cv::Mat homography = cv::findHomography(modelPoints,vprojVertices_bb);

            //warp current image in it
            cv::Mat patchCurr(box.size(), CV_8UC1);
            //cv::Mat &patchCurr = mNewINFO.NCCpatchScore;
            //patchCurr = cv::Mat(box.size(), CV_8UC1);

            //cv::warpPerspective( surf.mImage, patchCurr, homography, patchCurr.size() );
            //test filling with white to check if mask has effect we want
            //cv::warpPerspective( surf.mImage, patchCurr, homography, patchCurr.size(),INTER_LINEAR, BORDER_CONSTANT ,255);
            cv::warpPerspective( surf.mImage, patchCurr, homography, patchCurr.size(),INTER_LINEAR, BORDER_REPLICATE);

            //plot it on input
            patchCurr.copyTo(MDebugImgWarp(box));

            //compute NCC over search window
            //cv::Mat resultNCC = cv::Mat::zeros( window_search_size, window_search_size, CV_32FC1 );
            cv::Mat &resultNCC = mNewINFO.NCCpatchScore;
            //without taking care of image borders...
            //resultNCC = cv::Mat::zeros( window_search_size, window_search_size, CV_32FC1 );
            //cv::Rect myROI(box.x-half_window_size,box.y-half_window_size,box.size().width+window_search_size-1,box.size().height+window_search_size-1);
            
            //get the roi of the current image on which to compute similarity with patch
            //
            int myRoi_l = box.x-half_window_size; myRoi_l = (myRoi_l<0)?0:myRoi_l;
            int myRoi_t = box.y-half_window_size; myRoi_t = (myRoi_t<0)?0:myRoi_t;
            int myRoi_r = box.x+box.size().width+half_window_size; myRoi_r = (myRoi_r>img.size().width)?img.size().width:myRoi_r;
            int myRoi_d = box.y+box.size().height+half_window_size; myRoi_d = (myRoi_d>img.size().height)?img.size().height:myRoi_d;       

            cv::Rect myROI(myRoi_l,myRoi_t,myRoi_r-myRoi_l,myRoi_d-myRoi_t);//region of interest is around current position of point
            
            //verify that the search region is valid
            int result_cols = myROI.size().width - box.size().width + 1;
            int result_rows = myROI.size().height - box.size().height + 1;

            if(result_cols > half_window_size && result_rows > half_window_size)
            {
                resultNCC = cv::Mat::zeros( result_rows, result_cols, CV_32FC1 );

                cv::Mat imgNorm;
                std::cout<<v<<std::endl;
 //               if(v==4)
//                equalizeHist( MDebugImg(myROI), MDebugImg(myROI) );
//                equalizeHist( img(myROI), imgNorm );

//#ifdef USE_SIMIL_MAX
//                cv::matchTemplate( imgNorm, patchCurr, resultNCC, CV_TM_CCORR_NORMED, mask);
//#else
//                cv::matchTemplate( imgNorm, patchCurr, resultNCC, CV_TM_SQDIFF, mask);
//endif
#ifdef USE_SIMIL_MAX
                cv::matchTemplate( img(myROI), patchCurr, resultNCC, CV_TM_CCORR_NORMED, mask);
#else
                cv::matchTemplate( img(myROI), patchCurr, resultNCC, CV_TM_SQDIFF, mask);
#endif
                
                /// Localizing the best match with minMaxLoc searching for max NCC
                double minVal; double maxVal; cv::Point minLoc; cv::Point maxLoc;
                cv::minMaxLoc( resultNCC, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat() );

                //plot result NCC on input
                cv::Mat resultNCCnorm; resultNCC.copyTo(resultNCCnorm);
                cv::normalize( resultNCCnorm, resultNCCnorm, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );
                cv::Mat resultNCCuc( resultNCC.size().height, resultNCC.size().width, CV_8UC1);
                for(int i=0;i<resultNCC.size().height;i++)
                    for(int j=0;j<resultNCC.size().width;j++)
                        resultNCCuc.at<unsigned char>(i,j) = 255.*resultNCCnorm.at<float>(i,j);
                cv::Rect myROIncc(myROI.x + box.size().width/2,myROI.y + box.size().height/2,result_cols,result_rows);
                resultNCCuc.copyTo(MDebugImgScore(myROIncc));


                //keep maximum value location and whole score patch
                //so that can init PnP with a few maxVal locations and get corresponding NCC score
                //of the estimated poses
                mNewINFO.patchCoordinates = cv::Point(myROI.x + box.size().width/2,myROI.y + box.size().height/2);
#ifdef USE_SIMIL_MAX
                mNewINFO.relMaxLoc = maxLoc;
                cv::circle(MDebugImgScore,mNewINFO.getMaxLoc(),2,255);
#else
                mNewINFO.relMaxLoc = minLoc;
                cv::circle(MDebugImgScore,mNewINFO.getMaxLoc(),2,255);
#endif
                
            
                //std::cout<<"NCCmax patch = "<<maxVal<<std::endl;
                //std::cout<<"minVal = "<<minVal<<std::endl;
                //std::cout<<"NCCmax viewScore = "<<viewScore<<std::endl;
            }
        }
    }

    cv::imwrite("output/curr.png",MDebugImg);
    cv::imwrite("output/warp.png",MDebugImgWarp);
    //cv::imwrite("output/masks.png",MDebugImgMask);
    cv::imwrite("output/score.png",MDebugImgScore);
    
    std::cout<<"estimate best pose"<<std::endl;
    //perform PnPwith all subsets of 4 matches, estimate pose and check corresponding score
    //keep best scoring pose
    cv::Affine3d bestPose;
#ifdef USE_SIMIL_MAX
    float bestNCCscore=0;
#else
    float bestNCCscore=1e10;
#endif

    //do a kind of ransac: try different subset to compute pose util find that more than majority agrees
    //if not consider tracker lost
    const unsigned int nbBasePnp=3;//take four points out of set
    unsigned int pointers[nbBasePnp];
    for(unsigned int i=0;i<nbBasePnp;i++)pointers[i]=i;//set first pointers as first elements of list        

    while(1)
    {
        Vec3d rvec=prevPoseCam.rvec();
        Vec3d tvec=prevPoseCam.translation();
        
        //create vectors corresponding to subset
        vector<Point3f> subsetVertices;
        vector<Point2f> subsetProjections;
        for(unsigned int i=0;i<nbBasePnp;i++)
        {
            subsetVertices.push_back(mSearchInfo[pointers[i]].center);
            subsetProjections.push_back(mSearchInfo[pointers[i]].getMaxLoc());
        }
        
        
        //compute pose with subset
        cv::solvePnP(subsetVertices, subsetProjections, _mCalib.cameraMatrix, _mCalib.distCoeffs, rvec, tvec, true);

        //get corresponding NCC score for this pose
        //project all the surface centers onto image using new pose
        vector<Point3f> surfCenters;
        for(unsigned int v=0;v<mSearchInfo.size();v++)
            surfCenters.push_back(mSearchInfo[v].center);

        vector<Point2f> surfCenterProj;
        projectPoints(surfCenters, rvec, tvec, _mCalib.cameraMatrix, _mCalib.distCoeffs, surfCenterProj);

        //accumulate NCC scores
        //might have to change score accumulation as it penalizes the occlusions...
        //to allow for occlusion best might be to only accumulate the n best scores
        float NCCscore = 0;
        for(unsigned int v=0;v<mSearchInfo.size();v++)
        {
            NCCscore += mSearchInfo[v].getNCCscore(cv::Point(surfCenterProj[v]));
            //std::cout<<"Read score = "<<mSearchInfo[v].getNCCscore(cv::Point(surfCenterProj[v]))<<std::endl;
        }

        //if(NCCscore > bestNCCscore)
        if(NCCscore < bestNCCscore)
        {
            bestNCCscore = NCCscore;
            bestPose = cv::Affine3d(rvec,tvec);
        }
        //if not need to go to next subset:
        //try moving last pointer, if pointer reaches end, then have to move previous pointer
        //recursive if any pointer reaches its end position then moves it previous pointer and put current pointer after it
        //do until first point (pointers[0]) reaches the end (for pointers[0] corresponds to pointPositions.size-nbBasePnp)
        
        //to do that just count how many pointers are stuck at the end:
        bool nextSetAvailable = getNextSetPointers(&pointers[0],mSearchInfo.size(),nbBasePnp);

        if(!nextSetAvailable)
            break;        
    }

    std::cout<<"Best score = "<<bestNCCscore<<std::endl;
    poseCam = bestPose;

    //could refine by performing non linear optimization of score with respect to projection,
    //would then have to compute the gradients of all the NCC scores patches... 

}
*/

//surfaces matches to perform non linear optim and find 3D pose of the object
struct surfaceMatchInfo {
    cv::Point3f objCoord;//position of the center of surface in model
    cv::Point projCoord;
    
    float viewScore;//the more frontoparallel, the more it goes towards 1, goes to 0 otherwise
    float similarityScore;//the more similar, the higher (eg NCC or NCC * entropy)
} ;

void printMat(const CvMat* mat)
{
    //IplImage img=mat;
    cv::Mat matt=cvarrToMat(mat);
    std::cout<<mat->rows<<std::endl;
    std::cout<<mat->cols<<std::endl;
    std::cout<<matt.channels()<<std::endl;
    for(int i=0; i<mat->rows; i++)
    {
        for(int j=0; j<mat->cols; j++)
        {   
            for(int k=0; k<matt.channels(); k++)
                std::cout<<cvGet2D(mat,i,j).val[k]<<"\t";
            std::cout<<std::endl;
        }
        std::cout<<std::endl<<std::endl;
    }

}

//multiply matrix by a matrix with one element only
void mulErrMat(const CvMat* mat,const CvMat* s,CvMat* matr)
{
    //score has shape (1,nbMatch,1)
    //err has shape (1,nbMatch,2)
    cv::Mat mat_r=cvarrToMat(mat);
    cv::Mat mat_w=cvarrToMat(matr);
    int nbChannel = mat_r.channels();

    for(int i=0; i<mat->rows; i++)
    {
        //float *ptr_r = mat_r.at<float>
        for(int j=0; j<mat->cols; j++)
        {   
            for(int k=0; k<mat_r.channels(); k++)
                mat_w.ptr<double>(i)[j*nbChannel+k] = cvGet2D(s,i,j).val[0]*mat_r.ptr<double>(i)[j*nbChannel+k];
        }
    }

}

void mulJacMat(const CvMat* mat,const CvMat* s,CvMat* matr)
{
    //score has shape (1,nbMatch,1)
    //Jac has shape (2*nbMatch,6,1)
    cv::Mat mat_r=cvarrToMat(mat);
    cv::Mat mat_w=cvarrToMat(matr);

    for(int i=0; i<s->cols; i++)
        for(int i2=0; i2<2; i2++)
    {
        for(int j=0; j<mat->cols; j++)
        {   
            mat_w.ptr<double>(2*i+i2)[j] = cvGet2D(s,0,i).val[0]*mat_r.ptr<double>(2*i+i2)[j];
        }
    }

}

void cvFindPose( const CvMat* objectPoints,
                  const CvMat* imagePoints, const CvMat* scores, const CvMat* A,
                  const CvMat* distCoeffs, CvMat* rvec, CvMat* tvec)
{
    const int max_iter = 20;
    Ptr<CvMat> matM, _m, _mn, matL, _s;

    int count;
    double a[9], ar[9]={1,0,0,0,1,0,0,0,1};
    CvScalar Mc;
    double param[6];
    CvMat matA = cvMat( 3, 3, CV_64F, a );
    CvMat _Ar = cvMat( 3, 3, CV_64F, ar );
    CvMat _r = cvMat( 3, 1, CV_64F, param );
    CvMat _t = cvMat( 3, 1, CV_64F, param + 3 );
    CvMat _param = cvMat( 6, 1, CV_64F, param );
    CvMat _dpdr, _dpdt;

    CV_Assert( CV_IS_MAT(objectPoints) && CV_IS_MAT(imagePoints) &&
        CV_IS_MAT(A) && CV_IS_MAT(rvec) && CV_IS_MAT(tvec) );

    count = MAX(objectPoints->cols, objectPoints->rows);
    matM.reset(cvCreateMat( 1, count, CV_64FC3 ));
    _m.reset(cvCreateMat( 1, count, CV_64FC2 ));

    cvConvertPointsHomogeneous( objectPoints, matM );
    cvConvertPointsHomogeneous( imagePoints, _m );
    cvConvert( A, &matA );
    //cvConvert( scores, &_s );

    //std::cout<<"scores = "<<std::endl;
    //printMat(scores);


    CV_Assert( (CV_MAT_DEPTH(rvec->type) == CV_64F || CV_MAT_DEPTH(rvec->type) == CV_32F) &&
        (rvec->rows == 1 || rvec->cols == 1) && rvec->rows*rvec->cols*CV_MAT_CN(rvec->type) == 3 );

    CV_Assert( (CV_MAT_DEPTH(tvec->type) == CV_64F || CV_MAT_DEPTH(tvec->type) == CV_32F) &&
        (tvec->rows == 1 || tvec->cols == 1) && tvec->rows*tvec->cols*CV_MAT_CN(tvec->type) == 3 );

    _mn.reset(cvCreateMat( 1, count, CV_64FC2 ));

    // normalize image points
    // (unapply the intrinsic matrix transformation and distortion)
    cvUndistortPoints( _m, _mn, &matA, distCoeffs, 0, &_Ar );

    CvMat _r_temp = cvMat(rvec->rows, rvec->cols,
        CV_MAKETYPE(CV_64F,CV_MAT_CN(rvec->type)), param );
    CvMat _t_temp = cvMat(tvec->rows, tvec->cols,
        CV_MAKETYPE(CV_64F,CV_MAT_CN(tvec->type)), param + 3);
    cvConvert( rvec, &_r_temp );
    cvConvert( tvec, &_t_temp );
    
    

    cvReshape( matM, matM, 3, 1 );//3 = number of channels, 1=number of rows
    cvReshape( _mn, _mn, 2, 1 );

    // refine extrinsic parameters using iterative algorithm
    CvLevMarq solver( 6, count*2, cvTermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER,max_iter,FLT_EPSILON), true);
    cvCopy( &_param, solver.param );

    for(;;)
    {
        CvMat *matJ = 0, *_err = 0;
        const CvMat *__param = 0;
        bool proceed = solver.update( __param, matJ, _err );
        cvCopy( __param, &_param );
        if( !proceed || !_err )
            break;
        cvReshape( _err, _err, 2, 1 );

        //std::cout<<"_err = "<<std::endl;
        //printMat(_err);

        if( matJ )
        {
            cvGetCols( matJ, &_dpdr, 0, 3 );
            cvGetCols( matJ, &_dpdt, 3, 6 );
            cvProjectPoints2( matM, &_r, &_t, &matA, distCoeffs,
                              _err, &_dpdr, &_dpdt, 0, 0, 0 );
            //in _err just get the projection of the points now and 
        }
        else
        {
            cvProjectPoints2( matM, &_r, &_t, &matA, distCoeffs,
                              _err, 0, 0, 0, 0, 0 );
        }
        cvSub(_err, _m, _err);//now get the error really

        //std::cout<<"_err = "<<std::endl;
        //printMat(_err);
        //cvMul(_err, scores, _err);
        mulErrMat(_err, scores, _err);

        //std::cout<<"_errs = "<<std::endl;
        //printMat(_err);

        if( matJ )
        {
            //std::cout<<"matJ = "<<std::endl;
            //printMat(matJ);
            mulJacMat(matJ,scores,matJ);
            //std::cout<<"matJs = "<<std::endl;
            //printMat(matJ);
            //printMat(scores);
        }

        cvReshape( _err, _err, 1, 2*count );
        //printMat(_err);
    }
    cvCopy( solver.param, &_param );

    _r = cvMat( rvec->rows, rvec->cols,
        CV_MAKETYPE(CV_64F,CV_MAT_CN(rvec->type)), param );
    _t = cvMat( tvec->rows, tvec->cols,
        CV_MAKETYPE(CV_64F,CV_MAT_CN(tvec->type)), param + 3 );

    cvConvert( &_r, rvec );
    cvConvert( &_t, tvec );
}

//bool robustPnp(std::vector<cv::Point3f> objectPoints,std::vector<cv::Point> imagePoints,
//    std::vector<cv::Point3f> score, const IntrinsicCalibration &_mCalib, cv::Affine3d& poseCam)
bool robustPnp(InputArray opoints,InputArray ipoints,
//    std::vector<float> score, InputArray _cameraMatrix, InputArray _distCoeffs,
    InputArray score, InputArray _cameraMatrix, InputArray _distCoeffs,
               OutputArray _rvec, OutputArray _tvec)
{
    //undistort points 
    CvMat c_objectPoints = opoints.getMat(), c_imagePoints = ipoints.getMat();
    CvMat c_score = score.getMat();
    CvMat c_cameraMatrix = _cameraMatrix.getMat(), c_distCoeffs = _distCoeffs.getMat();
    CvMat c_rvec = _rvec.getMat(), c_tvec = _tvec.getMat();

    /*Mat Mscores(1,score.size(),CV_32FC1);
    for(int i=0;i<score.size();++i)
    {
        Mscores.at<float>(0,i)=score[i];
    }
    CvMat c_score = Mscores;*/
    
    cvFindPose(&c_objectPoints, &c_imagePoints, &c_score, &c_cameraMatrix,&c_distCoeffs,
                                     &c_rvec, &c_tvec);
    return true;
    
    //get list of MeasureTempPointfrom matches and best features
    /*std::vector<MeasureTempPoint> measurePoints;
    for(int m=0;m<matchesCurrent.size();m++)
    {
        int idFeaturep=matchesCurrent[m].i1p;
        //check if we got a 3D position estimation for the corresponding point
        std::vector<uptoscaleFeature>::iterator it = std::find_if (best_features.begin(), best_features.end(), CompareId(idFeaturep));
        if(it != best_features.end())
        {
            //match has a 3D position linked to its feature in reference
            MeasureTempPoint newMeasure;
            newMeasure.mMeasure=_myCamera->ToMeters(Vector2f(matchesCurrent[m].u1c,matchesCurrent[m].v1c));
            
            //measure in ref
            Vector2f mMeasureRef=_myCamera->ToMeters(Vector2f(matchesCurrent[m].u1p,matchesCurrent[m].v1p));
            newMeasure.mPosition=toHomogeneous(mMeasureRef)* it->depthInRef;
            measurePoints.push_back(newMeasure);
        }
        
    }
    
    if(measurePoints.size()>10)
    {
        //estimate relative pose by minimizing reprojection error:
        int nb_iter=20;
        
        for(int iter=0;iter<nb_iter;iter++)
        {

            VectorXf Jte(6);Jte.setZero();
            MatrixXf H(6,6);H.setZero();
            
            
            for(int i=0;i<measurePoints.size();i++)
            {
                //matrix to be filled           
                MeasureTempPoint measure=measurePoints[i];
                Vector3f mapPointsCam=relPose*measure.mPosition;
                
                //compute error with observed points (in meter in z=1 plane)
                Vector2f x_d=measure.mMeasure;//desired projection= measurement
                Vector2f x_c=_myCamera->ProjectZ1(mapPointsCam);//current projection
                
                Vector2f error=x_d-x_c;
                
                float norm_reproj_error=error.transpose()*error;

                //get jacobien  of error with respect to variation of camera pose
                MatrixXf de_dp=-_myCamera->ProjectZ1_Jac_Dp(mapPointsCam);  
                
                Jte+=de_dp.transpose()*error;
                
                H+=de_dp.transpose()*de_dp;
                
            }
            Eigen::FullPivLU<MatrixXf> lu(H);
            //todo try p+=Dp
            VectorXf Dp(6);
            Dp=-1.*(lu.inverse()*Jte);
            relPose=HomogeneousMatrix(Dp)* relPose;
        }
    }
    return relPose;*/
}

float MI(cv::Mat img, cv::Mat &templ, cv::Mat &mask)
{
    int nbBin = 8;
    float Pm[nbBin],Pn[nbBin],Pmn[nbBin*nbBin];

    //init bins
    for(int i=0;i<nbBin;i++)
    {
        Pm[i]=0;
        Pn[i]=0;
        for(int j=0;j<nbBin;j++)
            Pmn[nbBin*i+j]=0;
    }

    //fill bins
    //NN
    /*int nbPix = 0;
    for (int i = 0; i < img.rows; ++i)
    {
        uchar* pixel_mask = mask.ptr<uchar>(i);  // point to first color in row
        uchar* pixel_m = img.ptr<uchar>(i);  // point to first color in row
        uchar* pixel_n = templ.ptr<uchar>(i);  // point to first color in row
        for (int j = 0; j < img.cols; ++j)
        {

            uchar mask = *pixel_mask++;
            if (mask>0)
            {
                uchar m = *pixel_m++;
                uchar n = *pixel_n++;

                uchar bm = (int)(nbBin*(float)m/256.);
                uchar bn = (int)(nbBin*(float)n/256.);

                Pm[bm]++;
                Pn[bn]++;
                Pmn[nbBin*bm+bn]++;
                nbPix++;
            }
        }
    }*/

    int nbPix = 0;
    for (int i = 0; i < img.rows; ++i)
    {
        uchar* pixel_mask = mask.ptr<uchar>(i);  // point to first color in row
        uchar* pixel_m = img.ptr<uchar>(i);  // point to first color in row
        uchar* pixel_n = templ.ptr<uchar>(i);  // point to first color in row
        for (int j = 0; j < img.cols; ++j)
        {

            uchar mask = *pixel_mask++;
            if (mask>0)
            {
                uchar m = *pixel_m++;
                uchar n = *pixel_n++;

                float bm = (float)(nbBin-1.)*(float)m/256.;
                float bn = (float)(nbBin-1.)*(float)n/256.;

                int Ebm = (int)bm;
                int Ebn = (int)bn;
                float em = bm-(float)Ebm;
                float en = bn-(float)Ebn;

                Pm[Ebm]  += 1.- em;
                Pm[Ebm+1]+= em;

                Pn[Ebn]  += 1.- en;
                Pn[Ebn+1]+= en;

                Pmn[nbBin*(Ebm)+Ebn] += (1.- em)*(1.- en);
                Pmn[nbBin*(Ebm+1)+Ebn] += (em)*(1.- en);
                Pmn[nbBin*(Ebm)+Ebn+1] += (1.- em)*(en);
                Pmn[nbBin*(Ebm+1)+Ebn+1] += (em)*(en);

                nbPix++;
            }
        }
    }
    //normalise
    //int nbPix = img.rows*img.cols;
    for(int i=0;i<nbBin;i++)
    {
        Pm[i]=Pm[i]/nbPix;
        Pn[i]=Pn[i]/nbPix;
        for(int j=0;j<nbBin;j++)
            Pmn[nbBin*i+j]=Pmn[nbBin*i+j]/nbPix;
    }

    //compute MI
    float res = 0;
   for(int i=0;i<nbBin;i++)
    {
        for(int j=0;j<nbBin;j++)
            if(Pmn[nbBin*i+j]>0 && Pm[i]>0 && Pn[j]>0)
                res += Pmn[nbBin*i+j]*log(Pmn[nbBin*i+j]/(Pm[i]*Pn[j]));
    }



    return res;
}

void matchTemplateMI( cv::Mat img, cv::Mat &templ, cv::Mat &res, cv::Mat &mask)
{
    for(int i=0;i<res.size().height;i++)
        for(int j=0;j<res.size().width;j++)
        {
            Rect box = cv::Rect(j,i,templ.size().width,templ.size().height);
            res.at<float>(i,j) = MI(img(box),templ,mask);
        }
}

void parabolicRefinement(cv::Mat &curv,cv::Point maxLoc,cv::Point2f &maxLocF)
{
    if(maxLoc.x>0 && maxLoc.y>0 && maxLoc.x<curv.size().width-1 && maxLoc.y<curv.size().height-1)
        maxLocF = maxLoc;
    else
    {
        float centerVal = curv.at<float>(maxLoc.y,maxLoc.x);

        //fit parabola horizontally and vertically
        //for this simply get derivative x left and right and do linear regression
        float dx_l = centerVal - curv.at<float>(maxLoc.y,maxLoc.x-1);
        float dx_r = curv.at<float>(maxLoc.y,maxLoc.x+1) - centerVal;

        float dy_t = centerVal - curv.at<float>(maxLoc.y-1,maxLoc.x);
        float dy_b = curv.at<float>(maxLoc.y+1,maxLoc.x) - centerVal;

        float ex = -0.5 + dx_l / (dx_l-dx_r);
        float ey = -0.5 + dy_t / (dy_t-dy_b);

        maxLocF = cv::Point2f(maxLoc)+cv::Point2f(ex,ey);
    }
}               

void Object3D::track(const cv::Mat &img, const cv::Mat &prev_img, const IntrinsicCalibration &_mCalib, const cv::Affine3d& prevPoseCam, cv::Affine3d& poseCam) const
{
    //TODO: multiscale approach ? can simply scale image with respect to object distance
    //TODO: try template matching with other similarity functions: NCC has wide basin, more narrow optimum would be better for RANSAC

    //project textured planar surfaces to current image using the previous pose
    //and do NCC search to have 2D correspondance for all of them and then PnP
    //to retrieve the 3D pose from the sets of 2D matches



    cv::Mat MDebugImg;
    img.copyTo(MDebugImg); 
    cv::Mat MDebugImgWarp;
    img.copyTo(MDebugImgWarp);
    cv::Mat MDebugImgScore;
    img.copyTo(MDebugImgScore);
    cv::Mat MDebugImgScorer;
    img.copyTo(MDebugImgScorer);

    int window_search_size = 32;
    int half_window_size = window_search_size/2;

    int window_search_size_drift = 6;
    int half_window_size_drift = window_search_size_drift/2;

    //information to keep for PnP
    //std::vector<surfaceMatchInfo> mSearchInfo;
    std::vector<cv::Point3f> objectPoints;
    std::vector<cv::Point2f> imagePoints;
    std::vector<float> score;

    std::cout<<"compute NCC scores"<<std::endl;

    for(unsigned int v=0;v<mPlanarSurfaces.size();v++)
    {
        //std::cout<<"loop surf["<<v<<"]"<<std::endl;
        const planarSurface &surf = mPlanarSurfaces[v];

        //project surface corners to check if visible and get homography from image to surface model
        Point3f ptLine = prevPoseCam * pose * surf.center;
        Vec3d ray = Vec3d(ptLine.x,ptLine.y,ptLine.z);        ray = ray / norm(ray);
        Vec3d normal_cam = prevPoseCam.rotation() * pose.rotation() * surf.normal;
        //float viewScore = -normal_cam.dot(ray);
        float viewScore = 1.-2.*acos(-normal_cam.dot(ray))/3.141592;

        if(viewScore > 0.2)
        {
            vector<Point3f> ObjPoints;
            ObjPoints.push_back(surf.center+toPoint(surf.radius1*surf.b1+surf.radius2*surf.b2));
            ObjPoints.push_back(surf.center+toPoint(surf.radius1*surf.b1-surf.radius2*surf.b2));
            ObjPoints.push_back(surf.center+toPoint(-surf.radius1*surf.b1-surf.radius2*surf.b2));
            ObjPoints.push_back(surf.center+toPoint(-surf.radius1*surf.b1+surf.radius2*surf.b2));
            ObjPoints.push_back(surf.center);

            vector<Point3f> LineObj;
            for(int i=0;i<ObjPoints.size();i++) 
                LineObj.push_back(pose*ObjPoints[i]);
            
            //get position of surface corners in previous image
            vector<Point2f> vprojVertices;
            projectPoints(LineObj, prevPoseCam.rvec(), prevPoseCam.translation(), _mCalib.cameraMatrix, _mCalib.distCoeffs, vprojVertices);
            
            //get the minimal support rectangle which contains the projected surface
            Rect box = cv::boundingRect(cv::Mat(vprojVertices));

            //if entirely projects in image
            if(box.x >=0 && box.y >=0 && box.x + box.size().width < img.size().width && box.y + box.size().height < img.size().height)
            {
                surfaceMatchInfo newMatch;
                newMatch.objCoord = surf.center;
                newMatch.viewScore = viewScore;

                 //get projected corners coordinates in the bounding box
                vector<Point> vprojVertices_bb;
                for(int v=0;v<vprojVertices.size();v++)
                    vprojVertices_bb.push_back(vprojVertices[v]-Point2f(box.x, box.y));

                vector<Point2f> vprojVertices_bbF;
                for(int v=0;v<vprojVertices.size();v++)
                    vprojVertices_bbF.push_back(vprojVertices[v]-Point2f(box.x, box.y));


                //create the corresponding mask in box wheer surface is defined
                cv::Mat mask = cv::Mat::zeros( box.size(), CV_8UC1 );
                cv::fillConvexPoly(mask, &vprojVertices_bb[0], 4, 1);//4: use only 4 first corners in vertices to define contour, 5th is center

                //use previous image to define what we are searching for
                cv::Mat patchCurr(box.size(), CV_8UC1);
                prev_img(box).copyTo(patchCurr);

                //Debug : plot it on input
                patchCurr.copyTo(MDebugImgWarp(box));

                //compute NCC over search window
                //get the roi of the current image on which to compute similarity with patch
                int myRoi_l = box.x-half_window_size; myRoi_l = (myRoi_l<0)?0:myRoi_l;
                int myRoi_t = box.y-half_window_size; myRoi_t = (myRoi_t<0)?0:myRoi_t;
                int myRoi_r = box.x+box.size().width+half_window_size; myRoi_r = (myRoi_r>img.size().width)?img.size().width:myRoi_r;
                int myRoi_d = box.y+box.size().height+half_window_size; myRoi_d = (myRoi_d>img.size().height)?img.size().height:myRoi_d;       

                cv::Rect myROI(myRoi_l,myRoi_t,myRoi_r-myRoi_l,myRoi_d-myRoi_t);//region of interest is around current position of point
                
                //verify that the search region is valid
                int result_cols = myROI.size().width - box.size().width + 1;
                int result_rows = myROI.size().height - box.size().height + 1;

                if(result_cols > half_window_size && result_rows > half_window_size)
                //if(result_cols >= window_search_size && result_rows >= window_search_size)
                {
                    cv::Mat resultNCC = cv::Mat::zeros( result_rows, result_cols, CV_32FC1 );

#ifdef USE_SIMIL_MAX
                    cv::matchTemplate( img(myROI), patchCurr, resultNCC, CV_TM_CCORR_NORMED, mask);
#else
                    cv::matchTemplate( img(myROI), patchCurr, resultNCC, CV_TM_SQDIFF, mask);
#endif
                    
                    /// Localizing the best match with minMaxLoc searching for max NCC
                    double minVal; double maxVal; cv::Point minLoc; cv::Point maxLoc;
                    cv::minMaxLoc( resultNCC, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat() );

                    //plot result NCC on input
                    cv::Mat resultNCCnorm; resultNCC.copyTo(resultNCCnorm);
                    cv::normalize( resultNCCnorm, resultNCCnorm, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );
                    cv::Mat resultNCCuc( resultNCC.size().height, resultNCC.size().width, CV_8UC1);
                    for(int i=0;i<resultNCC.size().height;i++)
                        for(int j=0;j<resultNCC.size().width;j++)
                            resultNCCuc.at<unsigned char>(i,j) = 255.*resultNCCnorm.at<float>(i,j);
                    cv::Rect myROIncc(myROI.x + box.size().width/2,myROI.y + box.size().height/2,result_cols,result_rows);
                    resultNCCuc.copyTo(MDebugImgScore(myROIncc));
                    cv::circle(MDebugImgScore,minLoc + myROI.tl() + vprojVertices_bb[4],2,255);


                    //keep maximum value location and whole score patch
                    //so that can init PnP with a few maxVal locations and get corresponding NCC score
                    //of the estimated poses
#ifdef USE_SIMIL_MAX
                    newMatch.projCoord = maxLoc + myROI.tl() + vprojVertices_bb[4];
                    newMatch.similarityScore = maxVal;
#else
                    newMatch.projCoord = minLoc + myROI.tl() + vprojVertices_bb[4];
                    newMatch.similarityScore = -minVal;
#endif

                    //objectPoints.push_back(surf.center);
                    //imagePoints.push_back(newMatch.projCoord);
                    //score.push_back(viewScore*newMatch.similarityScore);

                    //now we have hopefully the position of the surface up to some drift
                    //so we want to use the model to correct the drift

                    //get homography from surface model to current image ROI
                    vector<Point2f> modelPoints;
                    modelPoints.push_back(Point2f(surf.mImage.size().width,0));
                    modelPoints.push_back(Point2f(surf.mImage.size().width,surf.mImage.size().height));
                    modelPoints.push_back(Point2f(0,surf.mImage.size().height));
                    modelPoints.push_back(Point2f(0,0));
                    modelPoints.push_back(Point2f(surf.mImage.size().width/2,surf.mImage.size().height/2));

                    cv::Mat homography = cv::findHomography(modelPoints,vprojVertices_bb);

                    cv::Mat patchCurr_drift(box.size(), CV_8UC1);
                    cv::warpPerspective( surf.mImage, patchCurr_drift, homography, patchCurr_drift.size(),INTER_LINEAR, BORDER_REPLICATE);
                    
                    //to make it simple, use box previously defined,
                    //box was centered on projection of vertex in previous image
                    //now center it on previously found optimum so can use same code afterward
#ifdef USE_SIMIL_MAX
                    box = box + maxLoc + myROI.tl() - box.tl();
                    //box = box + maxLoc - vprojVertices_bb[4];
#else
                    box = box + minLoc + myROI.tl() - box.tl();
                    //box = box + minLoc - vprojVertices_bb[4];
#endif

                    myRoi_l = box.x-half_window_size_drift; myRoi_l = (myRoi_l<0)?0:myRoi_l;
                    myRoi_t = box.y-half_window_size_drift; myRoi_t = (myRoi_t<0)?0:myRoi_t;
                    myRoi_r = box.x+box.size().width+half_window_size_drift; myRoi_r = (myRoi_r>img.size().width)?img.size().width:myRoi_r;
                    myRoi_d = box.y+box.size().height+half_window_size_drift; myRoi_d = (myRoi_d>img.size().height)?img.size().height:myRoi_d;       

                    myROI = cv::Rect(myRoi_l,myRoi_t,myRoi_r-myRoi_l,myRoi_d-myRoi_t);//region of interest is around current position of point
                    
                    //verify that the search region is valid
                    result_cols = myROI.size().width - box.size().width + 1;
                    result_rows = myROI.size().height - box.size().height + 1;

                    if(result_cols > half_window_size_drift && result_rows > half_window_size_drift)
                    {
                        cv::Mat resultNCC_drift = cv::Mat::zeros( result_rows, result_cols, CV_32FC1 );

                        //cv::matchTemplate( img(myROI), patchCurr_drift, resultNCC_drift, CV_TM_CCORR_NORMED, mask);
                        matchTemplateMI( img(myROI), patchCurr_drift, resultNCC_drift, mask);
                        //cv::matchTemplate( img(myROI), patchCurr_drift, resultNCC_drift, CV_TM_CCORR_NORMED);
                        
                        /// Localizing the best match with minMaxLoc searching for max NCC
                        cv::minMaxLoc( resultNCC_drift, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat() );

                        //refine maxLoc with parabolic fitting
                        cv::Point2f maxLocF;
                        parabolicRefinement(resultNCC_drift,maxLoc,maxLocF);


                        //plot result NCC on input
                        cv::Mat resultNCCnorm; resultNCC_drift.copyTo(resultNCCnorm);
                        cv::normalize( resultNCCnorm, resultNCCnorm, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );
                        cv::Mat resultNCCuc( resultNCC_drift.size().height, resultNCC_drift.size().width, CV_8UC1);
                        for(int i=0;i<resultNCC_drift.size().height;i++)
                            for(int j=0;j<resultNCC_drift.size().width;j++)
                                resultNCCuc.at<unsigned char>(i,j) = 255.*resultNCCnorm.at<float>(i,j);
                        cv::Rect myROIncc(myROI.x + box.size().width/2,myROI.y + box.size().height/2,result_cols,result_rows);
                        resultNCCuc.copyTo(MDebugImgScorer(myROIncc));

                        //print confidence
                        char confStr[100];
                        //sprintf(confStr, "%0.2f", viewScore);
                        sprintf(confStr, "%0.2f", maxVal);
                        //putText(MDebugImgScore, confStr,
                        putText(MDebugImgScorer, confStr,
                                    maxLoc + myROI.br(),
                                    cv::FONT_HERSHEY_COMPLEX_SMALL,
                                    0.8, cvScalar(250,250,250), 1, CV_AA);


                        //keep maximum value location and whole score patch
                        //newMatch.projCoord = maxLoc + myROI.tl() + vprojVertices_bb[4];
                        //newMatch.similarityScore = maxVal;
                        //mSearchInfo.push_back(newMatch);

                        objectPoints.push_back(surf.center);
                        imagePoints.push_back(maxLocF + Point2f(myROI.tl()) + vprojVertices_bbF[4]);
                        score.push_back(viewScore*maxVal);

                    }

                }
            }
        }
    }

    cv::imwrite("output/curr.png",MDebugImg);
    cv::imwrite("output/warp.png",MDebugImgWarp);
    //cv::imwrite("output/masks.png",MDebugImgMask);
    cv::imwrite("output/score.png",MDebugImgScore);
    cv::imwrite("output/scorer.png",MDebugImgScorer);
    
    std::cout<<"estimate best pose"<<std::endl;

    //poseCam = prevPoseCam;
    //robustPnp(mSearchInfo,_mCalib,poseCam);

    /*Vec3d rvec=prevPoseCam.rvec();
    Vec3d tvec=prevPoseCam.translation();

    std::cout<<objectPoints.size()<<std::endl;
    robustPnp(objectPoints,imagePoints,score,_mCalib.cameraMatrix, _mCalib.distCoeffs,rvec,tvec);
    poseCam = cv::Affine3d(rvec,tvec);*/


    //perform PnPwith all subsets of 4 matches, estimate pose and check corresponding score
    //keep best scoring pose
    cv::Affine3d bestPose;
    float bestScore=0;
    int bestValidSurf=0;

    //std::vector<cv::Point3f> objectPoints;
    //std::vector<cv::Point> imagePoints;
    //std::vector<float> score;
    bool inliers[objectPoints.size()];

    //do a kind of ransac: try different subset to compute pose util find that more than majority agrees
    //if not consider tracker lost
    const unsigned int nbBasePnp=4;//take four points out of set
    unsigned int pointers[nbBasePnp];
    for(unsigned int i=0;i<nbBasePnp;i++)pointers[i]=i;//set first pointers as first elements of list        

    while(1)
    {
        Vec3d rvec=prevPoseCam.rvec();
        Vec3d tvec=prevPoseCam.translation();
        
        //create vectors corresponding to subset
        vector<Point3f> subsetVertices;
        vector<Point2f> subsetProjections;
        for(unsigned int i=0;i<nbBasePnp;i++)
        {
            subsetVertices.push_back(objectPoints[pointers[i]]);
            subsetProjections.push_back(imagePoints[pointers[i]]);
        }
        
        
        //compute pose with subset
        cv::solvePnP(subsetVertices, subsetProjections, _mCalib.cameraMatrix, _mCalib.distCoeffs, rvec, tvec, true);

        //project all the surface centers onto image using new pose
        vector<Point3f> surfCenters;
        for(unsigned int v=0;v<objectPoints.size();v++)
            surfCenters.push_back(objectPoints[v] );

        vector<Point2f> surfCenterProj;
        projectPoints(surfCenters, rvec, tvec, _mCalib.cameraMatrix, _mCalib.distCoeffs, surfCenterProj);

        //check how many surfaces agree with the pose and get cumulative score
        int nbValidSurf = 0;
        float scoreAccu = 0;
        bool inliers_temp[objectPoints.size()];
        for(unsigned int v=0;v<imagePoints.size();v++)
        {
            if(cv::norm(surfCenterProj[v] - Point2f(imagePoints[v]))<5.)
            {
                nbValidSurf++;
                scoreAccu += score[v];
                inliers_temp[v] = true;
            }
            else
                inliers_temp[v] = false;
        }

        if(nbValidSurf > bestValidSurf)
        {
            bestValidSurf = nbValidSurf;
            bestScore = scoreAccu;

            bestPose = cv::Affine3d(rvec,tvec);

            //save inleirs
            for(unsigned int v=0;v<imagePoints.size();v++)
                inliers[v] = inliers_temp[v];

        }

        //if not need to go to next subset:
        bool nextSetAvailable = getNextSetPointers(&pointers[0],imagePoints.size(),nbBasePnp);

        if(!nextSetAvailable)
            break;        
    }
    std::cout<<"best cpt valid surf = "<<bestValidSurf<<std::endl;
    std::cout<<"score = "<<bestScore<<std::endl;
    
    //refine pose with inliers
    Vec3d rvec=prevPoseCam.rvec();
    Vec3d tvec=prevPoseCam.translation();
    
    //create vectors corresponding to subset
    vector<Point3f> subsetVertices;
    vector<Point2f> subsetProjections;
    vector<float> scores;
    for(unsigned int i=0;i<objectPoints.size();i++)
    {
        if(inliers[i])
        {
            subsetVertices.push_back(objectPoints[i]);
            subsetProjections.push_back(imagePoints[i]);
            scores.push_back(score[i]);
            //scores.push_back(1.);
        }
    }
      
    //compute pose with subset
    //cv::solvePnP(subsetVertices, subsetProjections, _mCalib.cameraMatrix, _mCalib.distCoeffs, rvec, tvec, true);
    robustPnp(subsetVertices, subsetProjections, scores, _mCalib.cameraMatrix, _mCalib.distCoeffs, rvec, tvec);

    poseCam = cv::Affine3d(rvec,tvec);

    //poseCam = bestPose;

    //could refine by performing non linear optimization of score with respect to projection,
    //would then have to compute the gradients of all the NCC scores patches... 

}

}

