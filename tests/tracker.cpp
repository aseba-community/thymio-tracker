
#include "arthymio.h"

#include "VideoSource.hpp"

static const char window_name[] = "Tracker";

int main(int argc, char** argv)
{
    videoSourceLive mVideoSource(EmbeddedCam);
    mVideoSource.resizeSource(0.5);
    
    cv::namedWindow( window_name, WINDOW_AUTOSIZE );
    
    cv::Mat outputImage;
    
    while(1)
    {
        mVideoSource.grabNewFrame();
        cv::Mat& inputImage = *mVideoSource.GetFramePointer();
        
        process(inputImage, outputImage);
        
        imshow(window_name, outputImage);
        
        if(waitKey(5) == 27)
            break;
    }
    
    return 0;
}
