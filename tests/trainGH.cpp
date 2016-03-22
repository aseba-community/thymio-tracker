//ModelViewer example: shows how to use Visualization3D
//and project model on any image

#include <iostream>
#include "Models.hpp"
#include "Visualization3D.hpp"
#include "GH.hpp"
#include "GHscale.hpp"


using namespace std;
using namespace cv;

namespace tt = thymio_tracker;

int main(int argc, const char * argv[])
{
    //create visualization tool
    Visualization3D vizu;
    tt::ThymioBlobModel mRobot;
    vizu.addObject(mRobot);
    
    //create an sphere of camera watching object
    vector<tt::Camera3dModel> vCams;
    float radiusSphere=0.3;//radius sphere
    //float distCamCam=0.25*radiusSphere;
    //float minLatitude=M_PI/6.;
    float distCamCam=0.15*radiusSphere;
    float minLatitude=M_PI/12.;
    
    /*float radiusSphere=0.3;//radius sphere
    float distCamCam=0.15*radiusSphere;
    float minLatitude=M_PI/12.;*/
    
    //get latitude angle increment from desired distCamCam
    int l=0;
    float latitude=M_PI/2.-l*distCamCam/radiusSphere;
    while(latitude>minLatitude)
    {
        if(l==0)//pole
        {
            //pole => want only one cam
            tt::Camera3dModel newCam;
            newCam.pose=newCam.pose.translate(Vec3d(0.0,0.0,-radiusSphere));
            
            vCams.push_back(newCam);
        }
        else
        {
            //compute how many cams we want at this latitude
            float radiusLatitude=2.*M_PI*cos(latitude);
            int nbCamInLat=(int)(radiusSphere*radiusLatitude/distCamCam);
            
            for(int cl=0;cl<nbCamInLat;cl++)
            {
                float longitude=2.*M_PI*cl/nbCamInLat;
                
                //rotate around z
                tt::Camera3dModel newCam;
                newCam.pose=newCam.pose.rotate(Vec3d(0.0,0.0,longitude));
                
                //rotate for latitude and zoom out
                Affine3d transfoRotx = Affine3d().rotate(Vec3d(M_PI/2.-latitude,0.0,0.0));
                newCam.pose=newCam.pose*transfoRotx;
               
                Affine3d transfoZLoc = Affine3d().translate(Vec3d(0.0,0.0,-radiusSphere));
                newCam.pose=newCam.pose*transfoZLoc;
                
                vCams.push_back(newCam);
            }
        }
        l++;
        latitude=M_PI/2.-l*distCamCam/radiusSphere;
    }
    
    for(unsigned int c=0;c<vCams.size();c++)vizu.addObject(vCams[c]);
    
    //create another window to show projection in created cameras
    char window_name[100] = "Camera Views";
    namedWindow( window_name, WINDOW_AUTOSIZE );
   
    //get projection matrix
    Mat cameraMatrix; Mat distCoeffs;
    FileStorage fs;    fs.open("/Users/amaurydame/Data/nexus/CamCalib/nexus_camera_calib.xml", FileStorage::READ);
    fs["camera_matrix"] >> cameraMatrix;fs["distortion_coefficients"] >> distCoeffs;
    Mat imBackground(405, 720, CV_8UC3, Scalar(0,0,0));
    tt::resizeCameraMatrix(cameraMatrix,Size(1920,1080),imBackground.size());
    moveWindow(window_name, 720, 0);

#define USE_SCALE
    
    //use created cameras to train GH to become robust to perspective
#ifndef USE_SCALE
    vector<Point2f> *projPoints=new vector<Point2f>[vCams.size()];//for GH: contains only coordinates in meters of projected vertices
#else
    vector<Point3f> *projPoints=new vector<Point3f>[vCams.size()];//for GHscale, contains inverse depth as well
#endif
    for(unsigned int p=0;p<vCams.size();p++)
    {
        Affine3d poseInv=vCams[p].pose.inv();
        //projPoints[p]=mRobot.projectVertices(cameraMatrix, distCoeffs, poseInv);//give coord in px, not good anymore
        
        Affine3d poseComb=poseInv * mRobot.pose;
        for(unsigned int v=0;v<mRobot.mVertices.size();v++)
        {
            Point3f pointCam=poseComb*mRobot.mVertices[v];
            //for GH:
#ifndef USE_SCALE
            Point2f projMeters(pointCam.x,pointCam.y);
#else
            //for GHscale:
            Point3f projMeters(pointCam.x/pointCam.z,pointCam.y/pointCam.z,1./pointCam.z);
#endif
            
            projPoints[p].push_back(projMeters);            
        }
        
        
    }
    
    //give that to GH
#ifndef USE_SCALE
    tt::GH mGH;//here will train with coodrinates in meters so calibration does not matter
    char GHfilename[100]="/Users/amaurydame/Projects/BlobotTracker/files/GH_Arth_Perspective.dat";
#else
    tt::GHscale mGH;//here will train with coodrinates in meters so calibration does not matter
    char GHfilename[100]="/Users/amaurydame/Projects/BlobotTracker/files/GHscale_Arth_Perspective.dat";
#endif
    
    mGH.initHashTable(mRobot.mVertices.size());
    mGH.setModel(projPoints,vCams.size());
    {
        //save GH for later use
        std::ofstream geomHashingStream(GHfilename, std::ios::out | std::ios::binary);
        mGH.saveToStream(geomHashingStream);
    }
    delete[] projPoints;
    
    //loop to switch from one cam to the next
    int key=0;
    int currentCam=0;
    while(key!=27)//esc press
    {
        cout<<"Display cam "<<currentCam<<endl;
        //display background image
        Mat imCam=imBackground.clone();
        
        //project object
        Affine3d poseInv=vCams[currentCam].pose.inv();
        mRobot.draw(imCam,cameraMatrix, distCoeffs, poseInv);
        imshow(window_name,imCam);
        
        //pass to next camera
        currentCam=(currentCam+1)%vCams.size();
        key = waitKey(0);
    }
    
    return 0;
}

