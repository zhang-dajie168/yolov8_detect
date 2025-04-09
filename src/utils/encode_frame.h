
#include <opencv2/opencv.hpp>
//using namespace cv;
#include <iostream>
#include <turbojpeg.h>
#include <vector>
#include <common.h>


// OpenCV Mat 编码为 JPEG（使用 TurboJPEG）
std::vector<uchar> tjpeg_encode(const cv::Mat& image, int quality = 95);

int  tjpeg_encode_frame(const cv::Mat& frame, image_buffer_t* image,int quality = 95);