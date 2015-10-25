#include "Models.hpp"

using namespace cv;
using namespace std;

Object3D::Object3D()
{}

void Object3D::draw(Mat& img, const Mat& cameraMatrix, const Mat& distCoeffs, const Affine3d& poseCam) const
{
    for(int e=0;e<mEdges.size();e++)
        drawEdge(mEdges[e], img, cameraMatrix, distCoeffs, poseCam);
    
    for(int v=0;v<mVertices.size();v++)
        drawVertice(mVertices[v], img, cameraMatrix, distCoeffs, poseCam);
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
    if(LineCam[0].z>0 && LineCam[1].z>0)
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
    for(int i=0;i<mMatches.size();i++)detectedVertices.push_back(mVertices[mMatches[i].id]);
    
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
    int nbBasePnp=4;//take four points out of set
    int pointers[nbBasePnp];
    for(int i=0;i<nbBasePnp;i++)pointers[i]=i;//set first pointers as first elements of list
    
    while(1)
    {
        //want to test all the unsorted subsets of 4 points
        Vec3d rvec,tvec;
        if(!init)    {rvec=rvec_prec;tvec=tvec_prec;}
        
        //create vectors corresponding to subset
        vector<Point3f> subsetVertices;
        vector<Point2f> subsetProjections;
        for(int i=0;i<nbBasePnp;i++)
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
            float threshold_proj=20.;//set error max to 15 pixels
            int nbPointAgree=0;
            vector<Point2f> vProjPoints;//project all points
            projectPoints(detectedVertices, rvec, tvec, _mCalib.cameraMatrix, _mCalib.distCoeffs, vProjPoints);
            for(int i=0;i<mMatches.size();i++)
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
                for(int i=0;i<mMatches.size();i++)
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
            for(int i=nbBasePnp-1-nbPtStuckAtEnd+1;i<nbBasePnp;i++)
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

ArthymioBlobModel::ArthymioBlobModel()
{
    mVertices.push_back(Point3f(0,5.6,0));
    mVertices.push_back(Point3f(0,4.8,0));
    mVertices.push_back(Point3f(0,0.8,0));
    mVertices.push_back(Point3f(0,0,0));
    
    mVertices.push_back(Point3f(0.8,4.8,0));
    mVertices.push_back(Point3f(0.8,0.8,0));
    mVertices.push_back(Point3f(0.8,0,0));
    
    mVertices.push_back(Point3f(8.,4.8,0));
    mVertices.push_back(Point3f(8.,0.8,0));
    mVertices.push_back(Point3f(8.,0,0));
    
    mVertices.push_back(Point3f(8.8,5.6,0));
    mVertices.push_back(Point3f(8.8,4.8,0));
    mVertices.push_back(Point3f(8.8,0.8,0));
    mVertices.push_back(Point3f(8.8,0,0));
    
    //set them in meter and center them
    for(int v=0;v<14;v++)
        mVertices[v]=(mVertices[v]-Point3f(4.4,2.8,0))/100.;
    
    //if want to display edges
    vector<Point3f> mVerticesTemp;
    //create vertices from dimensions
    mVerticesTemp.push_back(Point3f(-0.65,7.75,0));
    mVerticesTemp.push_back(Point3f(-0.65,-0.65,0));
    mVerticesTemp.push_back(Point3f(9.5,-0.65,0));
    mVerticesTemp.push_back(Point3f(9.5,7.75,0));
    //set them in meter and center them
    for(int v=0;v<4;v++)
        mVerticesTemp[v]=(mVerticesTemp[v]-Point3f(4.4,2.8,0))/100.;
    
    //from center to image plane
    mEdges.push_back(ModelEdge(mVerticesTemp[0],mVerticesTemp[1]));
    mEdges.push_back(ModelEdge(mVerticesTemp[1],mVerticesTemp[2]));
    mEdges.push_back(ModelEdge(mVerticesTemp[2],mVerticesTemp[3]));

}
