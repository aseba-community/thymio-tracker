
#include "arthymio.h"

#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
// #include <opencv2/calib3d.hpp>

// namespace basic_model
// {

// const cv::Point2f points[] = {
//     cv::Point2f(-4.4150390625, -3.56640625),
//     cv::Point2f(-4.3935546875, 1.998046875),
//     cv::Point2f(4.3291015625, -3.587890625),
//     cv::Point2f(4.33984375, 1.9765625)
// //    {-4.4150390625, -3.56640625}, {-4.3935546875, 1.998046875}, {4.3291015625, -3.587890625}, {4.33984375, 1.9765625}
// };

// const cv::Vec4f lines[] = {
//     cv::Vec4f(0.00289078, 0.999996, -4.97363, -1.60059),
//     cv::Vec4f(0.0029145, 0.999996, 4.91992, -1.58447),
//     cv::Vec4f(0.999997, -0.00244387, -0.0161133, 2.57812)
// //    {0.00289078, 0.999996, -4.97363, -1.60059}, {0.0029145, 0.999996, 4.91992, -1.58447}, {0.999997, -0.00244387, -0.0161133, 2.57812}
// };

// }

// Cartesian product of two vectors
template<class T1, class T2>
void product(const std::vector<T1>& v1, const std::vector<T2>& v2,
             std::vector<T1>& o1, std::vector<T2>& o2)
{
    typename std::vector<T1>::const_iterator it1 = v1.begin();
    const typename std::vector<T1>::const_iterator it1_end = v1.end();
    o1.clear();
    o2.clear();
    
    for(; it1 != it1_end; ++it1)
    {
        typename std::vector<T2>::const_iterator it2 = v2.begin();
        const typename std::vector<T2>::const_iterator it2_end = v2.end();
        for(; it2 != it2_end; ++it2)
        {
            o1.push_back(*it1);
            o2.push_back(*it2);
        }
    }
}

static cv::Mat gray;
static cv::Mat reduced;

static cv::Ptr<cv::LineSegmentDetector> lsd = cv::createLineSegmentDetector();
static cv::Ptr<cv::SimpleBlobDetector> sbd;

static const double scale = 0.5;

int process(const cv::Mat& input, cv::Mat& output)
{
    std::vector<cv::Vec4i> lines;
    std::vector<cv::KeyPoint> blobs;
    
    if(sbd.empty())
    {
        cv::SimpleBlobDetector::Params params;
        params.thresholdStep = 10;
        params.minThreshold = 100;
        params.maxThreshold = 200;
        params.minDistBetweenBlobs = 1;
        params.minRepeatability = 3;
        
        params.filterByColor = true;
        params.blobColor = 0;
        
        params.filterByArea = true;
        params.minArea = 5;
        params.maxArea = 800;
        
        params.filterByCircularity = true;
        params.minCircularity = 0.6;
        params.maxCircularity = 1.4;
        
        params.filterByInertia = false;
        params.minInertiaRatio = 0.2;
        params.maxInertiaRatio = 1.8;
        
        params.filterByConvexity = false;
        sbd = cv::SimpleBlobDetector::create(params);
    }
    
    cv::resize(input, reduced, cv::Size(0, 0), scale, scale);
    cv::cvtColor(reduced, gray, CV_RGB2GRAY);
    
    // Line and blob detector
    // lsd->detect(gray, lines);
    sbd->detect(gray, blobs);
    
    // Draw procedure
    input.copyTo(output);
    
    if(blobs.size() == 0)
        return 0;
    
    // Draw lines
    std::vector<cv::Vec4i>::const_iterator line_it = lines.begin();
    const std::vector<cv::Vec4i>::const_iterator line_it_end = lines.end();
    for(; line_it != line_it_end; ++line_it)
    {
        const cv::Vec4i& line = *line_it;
        cv::Point p1(line[0], line[1]);
        cv::Point p2(line[2], line[3]);
        cv::line(output, p1 / scale, p2 / scale, cv::Scalar(0, 0, 255), 2);
    }
    
    // Draw blobs
    // cv::drawKeypoints(output, blobs, output);
    std::vector<cv::KeyPoint>::const_iterator blob_it = blobs.begin();
    const std::vector<cv::KeyPoint>::const_iterator blob_it_end = blobs.end();
    for(; blob_it != blob_it_end; ++blob_it)
    {
        const cv::KeyPoint& keypoint = *blob_it;
        cv::circle(output, keypoint.pt / scale, (keypoint.size / scale - 1) / 2 + 1, cv::Scalar(255, 0, 0), -1);
        // cv::line(output, keypoint.pt / scale, keypoint.pt / scale, cv::Scalar(0, 255, 0), 1);
    }
    
    return 0;
}

int get_rows(const cv::Mat& m)
{
    return m.rows;
}
