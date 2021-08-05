
#ifndef WATERMARKCOREC___DCT_WATERMARK_H
#define WATERMARKCOREC___DCT_WATERMARK_H

#include <iostream>
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <string>
#include <algorithm>
#include <cstdlib>

class DCT_Watermark
{
private:
  cv::Mat Height_Original;
  cv::Mat Width_Original;
  cv::Mat ConvertedImage;
  cv::Mat Height;
  cv::Mat Width;
public:
  void embed(const cv::Mat& image, cv::Mat Watermark);
  void extract(cv::Mat image, cv::Mat Watermark);
};


#endif //WATERMARKCOREC___DCT_WATERMARK_H
