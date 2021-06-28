#include "Preprocess.h"

void Preprocess::ReadImageFromFile()
{
  cv::String file_location = "/Users/arkia/ComputerScienceRelated/Watermark Faker/Watermark Faker Data/facades/test";
  std::vector<cv::String> filenames;
  //Save the filename into the vector
  cv::glob(file_location, filenames);
  for (size_t looptimes = 0; looptimes < filenames.size(); looptimes++)
  {
    std::cout << filenames.at(looptimes) << std::endl;
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
  cv::Rect CropArea(0, 0, 256, 256); //����Ϊ256���ص�ͼƬ
  for (size_t looptimes = 0; looptimes < image_Readin.size(); looptimes++)
  {
    cv::Mat image_region = image_Readin.at(looptimes);
    image_Cropped.push_back(image_region(CropArea));
    //չʾ�ü�������Ƭ
    cv::imshow("Image", image_Cropped);
    cv::waitKey(0);
  }
}

void Preprocess::CropImageRightSide()
{
  cv::Rect CropArea(256, 256, 256, 256); //����Ϊ256���ص�ͼƬ
  for (size_t looptimes = 0; looptimes < image_Readin.size(); looptimes++)
  {
    cv::Mat image_region = image_Readin.at(looptimes);
    image_Cropped.push_back(image_region(CropArea));
    //չʾ�ü�������Ƭ
    cv::imshow("Image", image_Cropped);
    cv::waitKey(0);
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
    outfile << NormalizedImage.at(loop);
  }

  outfile.close();
}