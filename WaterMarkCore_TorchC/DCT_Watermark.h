
#ifndef WATERMARKCOREC___DCT_WATERMARK_H
#define WATERMARKCOREC___DCT_WATERMARK_H

#include <iostream>
#include <torch/torch.h>
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <string>
#include <algorithm>
#include <cstdlib>

class DCT_Watermark
{
private:
  std::vector<cv::String> SourceNames;
  std::vector<cv::String> WatermarkNames;
  cv::Mat SourceFile;
  cv::Mat WatermarkFile;
  cv::Mat DCT;
  cv::Mat Output;
  std::string source_path = "/Users/arkia/ComputerScienceRelated/Watermark Faker/Watermark Faker Data/DCT";
  std::string watermark_path = "/Users/arkia/ComputerScienceRelated/Watermark Faker/Watermark Faker Data/Watermark";
  std::string output_path = "/Users/arkia/ComputerScienceRelated/Watermark Faker/Watermark Faker Data/Output/Watermarked";
public:
  void ReadImageFromDisk();
  void Generator();
  void GetCSVFromDisk();
  void CoreTensorModule();
  void CheckpointOutput();
  
};


#endif //WATERMARKCOREC___DCT_WATERMARK_H
