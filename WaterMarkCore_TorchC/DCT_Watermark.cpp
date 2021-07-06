//
// Created by AKRIA on 2021/7/5.
//

#include "DCT_Watermark.h"

void DCT_Watermark::ReadImageFromDisk()
{
  cv::glob(watermark_path,WatermarkNames);
  cv::glob(source_path,SourceNames);
  for(auto & SourceName : SourceNames)
  {
    SourceFile = cv::imread(SourceName,cv::IMREAD_GRAYSCALE);
  }
  for(auto & WatermarkName : WatermarkNames)
  {
    WatermarkFile = cv::imread(WatermarkName,cv::IMREAD_GRAYSCALE);
  }
}

void DCT_Watermark::Generator()
{
  cv::dct(SourceFile,DCT,true);
  cv::addWeighted(WatermarkFile,0,DCT,1,0.0,Output);
  cv::imwrite(output_path,Output);
}

void DCT_Watermark::GetCSVFromDisk()
{
   std::ifstream infile;
   infile.open("D:/CS-Related/Watermark Faker/PreprocessDir/Preprocess_B_Image.csv");
   
}

void DCT_Watermark::CoreTensorModule()
{

}

void DCT_Watermark::CheckpointOutput()
{
  
}