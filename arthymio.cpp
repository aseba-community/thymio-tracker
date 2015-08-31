
#include "arthymio.h"

#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

static cv::Mat gray;
static cv::Mat reduced;
static cv::Mat edges;

static cv::Ptr<cv::LineSegmentDetector> lsd = cv::createLineSegmentDetector();

static cv::Ptr<cv::SimpleBlobDetector> sbd;

int process(const cv::Mat& input, cv::Mat& output)
{
    std::vector<cv::Vec4i> lines;
    std::vector<cv::KeyPoint> blobs;
    
    if(sbd.empty())
    {
        cv::SimpleBlobDetector::Params params;
        params.thresholdStep = 10;
        params.minThreshold = 100;
        params.maxThreshold = 150;
        params.minDistBetweenBlobs = 2;
        params.minRepeatability = 3;
        
        params.filterByColor = true;
        params.blobColor = 0;
        
        params.filterByArea = true;
        params.minArea = 4;
        params.maxArea = 200;
        
        params.filterByCircularity = true;
        params.minCircularity = 0.8;
        params.maxCircularity = 1.2;
        
        params.filterByInertia = false;
        params.minInertiaRatio = 0.2;
        params.maxInertiaRatio = 1.8;
        
        params.filterByConvexity = false;
        sbd = cv::SimpleBlobDetector::create(params);
    }
    
    cv::resize(input, reduced, cv::Size(320, 240), 0, 0);
    cv::cvtColor(reduced, gray, CV_RGB2GRAY);
    
    // Line and blob detector
    lsd->detect(gray, lines);
    sbd->detect(gray, blobs);
    
    // cv::cvtColor(edges, output, CV_GRAY2RGB);
    reduced.copyTo(output);
    
    // Draw lines
    std::vector<cv::Vec4i>::const_iterator line_it = lines.begin();
    const std::vector<cv::Vec4i>::const_iterator line_it_end = lines.end();
    for(; line_it != line_it_end; ++line_it)
    {
        const cv::Vec4i& line = *line_it;
        cv::Point p1(line[0], line[1]);
        cv::Point p2(line[2], line[3]);
        cv::line(output, p1, p2, cv::Scalar(0, 0, 255), 1);
    }
    
    // Draw blobs
    std::vector<cv::KeyPoint>::const_iterator blob_it = blobs.begin();
    const std::vector<cv::KeyPoint>::const_iterator blob_it_end = blobs.end();
    for(; blob_it != blob_it_end; ++blob_it)
    {
        const cv::KeyPoint& keypoint = *blob_it;
        cv::circle(output, keypoint.pt, (keypoint.size - 1) / 2 + 1, cv::Scalar(255, 0, 0), -1);
    }
    
    return 0;
}

int get_rows(const cv::Mat& m)
{
    return m.rows;
}
