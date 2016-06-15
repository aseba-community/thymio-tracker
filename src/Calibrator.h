
#ifndef _CALIBRATOR_H
#define _CALIBRATOR_H

#include <string>
#include <sstream>
#include <ctime>



#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/video.hpp>

namespace thymio_tracker
{


class Calibrator
{
public:
    Calibrator(const std::string& calibFileName);
    Calibrator(cv::FileStorage* calibFileName);
    void init(cv::FileStorage* calibStorage);

    //process a new frame: try detecting calibration pattern...
    void update(const cv::Mat& input);

    void drawState(cv::Mat* output) const;

private:
    bool calibrated;
    int cpt_found_in_a_row;
    uint nb_required_images;

    //pattern model
    std::vector<cv::Point3f> objectCorners;

    //list of list of 3d -> 2d amtches
    std::vector<std::vector<cv::Point3f> > objectPoints;//3d vertices
    std::vector<std::vector<cv::Point2f> > imagePoints;//projection of 3d vertices

    //where to store the calibration xml file
    cv::FileStorage fs;
    cv::FileStorage *fsPtr;

    //for display
    bool found;//pattern found in current frame
    std::vector<cv::Point2f> pointBuf;


};

}

#endif // 
