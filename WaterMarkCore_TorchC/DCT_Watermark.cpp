#include "DCT_Watermark.h"


void DCT_Watermark::embed(const cv::Mat& image, cv::Mat Watermark)
{
  Height_Original = image.rows;
  Width_Original = image.cols;
  image.convertTo (ConvertedImage,CV_32FC1);
  Height = ConvertedImage.rows;
  Width = ConvertedImage.cols;
  Watermark = nc::round(Watermark / 255);
  for(size_t heighloop = 0; heighloop < Height.size(); heighloop++)
  {
  
  }
}

void DCT_Watermark::extract(cv::Mat image, cv::Mat Watermark)
{


}

