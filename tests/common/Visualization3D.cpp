#include "Visualization3D.hpp"


bool ButtonLDown=false;
int posx_prec,posy_prec;

void CallBackFuncVizu3d(int event, int x, int y, int flags, void* userdata)
{
    Visualization3D *mVisualizer=(Visualization3D*)userdata;
    //Affine3d transform = Affine3d().rotate(Vec3d(0.00, 0.0, 0.01)).translate(Vec3d(0.0, 0.0, 0.0));
    if ( event == EVENT_LBUTTONDOWN )
    {
        ButtonLDown=true;
        posx_prec=x;posy_prec=y;
    }
    else if ( event == EVENT_LBUTTONUP)
    {
        ButtonLDown=false;
    }

    if(ButtonLDown && (event == EVENT_MOUSEMOVE))
    {
        int dx=x-posx_prec;
        int dy=y-posy_prec;
        
        if (flags == EVENT_FLAG_ALTKEY + EVENT_FLAG_LBUTTON)//translate on x and y
        {
            mVisualizer->cameraPose=mVisualizer->cameraPose.translate(Vec3d(-0.005*dx, 0.00 , 0.00));
            mVisualizer->cameraPose=mVisualizer->cameraPose.translate(Vec3d(0.00, -0.005*dy, 0.00));
        }
        else if(flags == EVENT_FLAG_SHIFTKEY + EVENT_FLAG_LBUTTON)//translate on z
        {
            mVisualizer->cameraPose=mVisualizer->cameraPose.translate(Vec3d(0.00, 0.00, 0.005*dy));
        }
        else //rotate around x and y
        {
            mVisualizer->cameraPose=mVisualizer->cameraPose.rotate(Vec3d(0.0, -0.005*dx, 0.00));
            mVisualizer->cameraPose=mVisualizer->cameraPose.rotate(Vec3d(0.005*dy, 0.0, 0.00));
        }
        mVisualizer->draw();
        
        posx_prec=x;
        posy_prec=y;
        
    }
    
}
Visualization3D::Visualization3D()
{
    //set intrinsic camera parameters, load it for now
    FileStorage fs;    fs.open("/Users/amaurydame/Data/nexus/CamCalib/nexus_camera_calib.xml", FileStorage::READ);
    fs["camera_matrix"] >> cameraMatrix;fs["distortion_coefficients"] >> distCoeffs;

    //create visualization window
    windowName = "Visualization 3D";
    
    //init position of camera
    //Affine3d cloud_pose = Affine3d().rotate(Vec3d(0.0, CV_PI/2, 0.0)).rotate(Vec3d(0.0, 0.0, CV_PI)).translate(Vec3d(0.0, 0.0, 3.0));
    cameraPose = Affine3d().rotate(Vec3d(CV_PI/3, 0.0, 0.0)).rotate(Vec3d(0.0, 0.0, 0.0)).translate(Vec3d(0.0, 0.0, 0.7));
    
    //create background
    imBackground= Mat(405, 720, CV_8UC3, Scalar(0,0,0));
    resizeCameraMatrix(cameraMatrix,Size(1920,1080),imBackground.size());
    
    //create window
    namedWindow( windowName, WINDOW_AUTOSIZE );
    setMouseCallback(windowName, CallBackFuncVizu3d, (void*)this);
    moveWindow(windowName, 0, 0);
    
}

void Visualization3D::addObject(Object3D &_obj)
{
    mObjects.push_back(_obj);
    draw();
}


void Visualization3D::draw()
{
    //clean image
    imCurrent=imBackground.clone();
    
    //draw objects
    for(int o=0;o<mObjects.size();o++)mObjects[o].draw(imCurrent,cameraMatrix, distCoeffs, cameraPose);
    imshow(windowName,imCurrent);

}

Visualization3D::~Visualization3D()
{
    destroyWindow(windowName);
}

