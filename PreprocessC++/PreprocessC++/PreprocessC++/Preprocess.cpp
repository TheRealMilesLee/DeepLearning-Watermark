#include "Preprocess.h"

void Preprocess::ReadImageFromFile()
{
  cv::String file_location = "D:\\CS-Related\\Watermark Faker\\Test_Images\\facades\\test";
  std::vector<cv::String> filenames;
  //Save the filename into the vector
  cv::glob(file_location, filenames);
  for (size_t looptimes = 0; looptimes < filenames.size(); looptimes++)
  {
    cv::Mat src = cv::imread(filenames.at(looptimes));

    if (!src.data)
    {
      std::cerr << "Problem loading image!!!" << std::endl;
    }
    image_Readin.push_back(src);
  }
}

void Preprocess::CropImageLeftSide()
{
  cv::Rect CropArea(0, 0, 256, 256); //²ÃÇÐÎª256ÏñËØµÄÍ¼Æ¬
  for (size_t looptimes = 0; looptimes < image_Readin.size(); looptimes++)
  {
    cv::Mat imageCropTool = cv::Mat(image_Readin.at(looptimes), CropArea);
    image_Cropped.at(looptimes).push_back(imageCropTool.clone());
  }
}

void Preprocess::CropImageRightSide()
{
  cv::Rect CropArea(256, 256, 256, 256); //²ÃÇÐÎª256ÏñËØµÄÍ¼Æ¬
  for (size_t looptimes = 0; looptimes < image_Readin.size(); looptimes++)
  {
    cv::Mat imageCropTool = cv::Mat(image_Readin.at(looptimes), CropArea);
    cv::Mat CroppedImg = imageCropTool.clone();
    image_Cropped.at(looptimes).push_back(CroppedImg);
  }
}

void Preprocess::ConvergentImage()
{
  int LowerBound = 0;
  int UpperBound = 1;
  cv::normalize(image_Cropped, NormalizedImage, LowerBound, UpperBound, cv::NORM_MINMAX);
}

void Preprocess::OutputAsCSV()
{
  std::ofstream outfile;
  outfile.open("D:\\CS-Related\\Watermark Faker\\PreprocessDir\\Preprocess.csv");
  for (size_t loop = 0; loop < NormalizedImage.size(); loop++)
  {
    OutputVector.at(loop) = NormalizedImage.at(loop);
    outfile << OutputVector.at(loop);
  }

  outfile.close();
}