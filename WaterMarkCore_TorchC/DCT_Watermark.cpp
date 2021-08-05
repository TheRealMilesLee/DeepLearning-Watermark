#include "DCT_Watermark.h"


void DCT_Watermark::embed(const cv::Mat& image, cv::Mat Watermark)
{
  Height_Original = image.rows;
  Width_Original = image.cols;
  image.convertTo (ConvertedImage,CV_32FC1);
  Height = ConvertedImage.rows;
  Width = ConvertedImage.cols;
  
}

void DCT_Watermark::extract(cv::Mat image, cv::Mat Watermark)
{


}

