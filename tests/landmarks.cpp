
#include "VideoSource.hpp"

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

static const char window_name[] = "Landmarks";

namespace tt = thymio_tracker;

int main(int argc, char** argv)
{
    VideoSourceLive videoSource(EmbeddedCam);
    videoSource.resizeSource(0.5);
    
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
    cv::Mat outputImage;
    
    cv::Ptr<cv::ORB> train_orb = cv::ORB::create(500, 1.2f, 8);
    cv::Ptr<cv::ORB> test_orb = cv::ORB::create(1000);
    // cv::Ptr<cv::BRISK> train_orb = cv::BRISK::create();
    // cv::Ptr<cv::BRISK> test_orb = cv::BRISK::create();
    
    std::vector<cv::KeyPoint> detectedKeypoints;
    std::vector<cv::KeyPoint> templateKeypoints;
    cv::Mat detectedDescriptors;
    cv::Mat templateDescriptors;
    
    // cv::Mat templateImage = cv::imread("../data/landmarks/oack_10_gray.png");
    cv::Mat templateImage = cv::imread("../data/landmarks/hut_30_gray.png");
    train_orb->detectAndCompute(templateImage, cv::noArray(), templateKeypoints, templateDescriptors);
    
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch> > matches;
    // matcher.add(templateDescriptors);
    
    while(true)
    {
        videoSource.grabNewFrame();
        const cv::Mat& inputImage = videoSource.getFramePointer();
        
        test_orb->detectAndCompute(inputImage, cv::noArray(), detectedKeypoints, detectedDescriptors);
        
        matches.clear();
        matcher.knnMatch(detectedDescriptors, templateDescriptors, matches, 2);
        
        std::vector<std::vector<char> > goodMatches;
        for(auto el : matches)
        {
            if(el[0].distance < 0.75 * el[1].distance)
                goodMatches.push_back({1, 0});
            else
                goodMatches.push_back({0, 0});
        }
        
        // inputImage.copyTo(outputImage);
        // cv::drawKeypoints(inputImage, detectedKeypoints, outputImage);
        cv::drawMatches(inputImage, detectedKeypoints,
                        templateImage, templateKeypoints,
                        matches, outputImage,
                        cv::Scalar(0, 0, 255),
                        cv::Scalar(255, 0, 0),
                        goodMatches);
        
        imshow(window_name, outputImage);
        
        auto key = cv::waitKey(5);
        if(key == 27 || key == 'q')
            break;
    }
    
    return 0;
}
