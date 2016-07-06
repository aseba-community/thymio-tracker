#include "Visualization3D.hpp"

using namespace std;
using namespace cv;

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
Visualization3D::Visualization3D(tt::IntrinsicCalibration *_mPtrCalibration)
{
    //create visualization window
    windowName = "Visualization 3D";
    
    //init position of camera
    //Affine3d cloud_pose = Affine3d().rotate(Vec3d(0.0, CV_PI/2, 0.0)).rotate(Vec3d(0.0, 0.0, CV_PI)).translate(Vec3d(0.0, 0.0, 3.0));
    cameraPose = Affine3d().rotate(Vec3d(CV_PI/3, 0.0, 0.0)).rotate(Vec3d(0.0, 0.0, 0.0)).translate(Vec3d(0.0, 0.0, 0.7));
    
    //create background
    mPtrCalibration = _mPtrCalibration;
     imBackground= Mat(mPtrCalibration->imageSize.height, mPtrCalibration->imageSize.width, CV_8UC3, Scalar(0,0,0));
    
    //create window
    namedWindow( windowName, WINDOW_AUTOSIZE );
    setMouseCallback(windowName, CallBackFuncVizu3d, (void*)this);
    moveWindow(windowName, 0, 0);
    
}

void Visualization3D::addObject(tt::Object3D &_obj)
{
    mObjects.push_back(_obj);
    draw();
}


void Visualization3D::draw()
{
    //clean image
    imCurrent=imBackground.clone();
    
    //draw objects
    for(unsigned int o=0;o<mObjects.size();o++)mObjects[o].draw(imCurrent,mPtrCalibration->cameraMatrix, mPtrCalibration->distCoeffs, cameraPose);
    imshow(windowName,imCurrent);

}

Visualization3D::~Visualization3D()
{
    destroyWindow(windowName);
}

