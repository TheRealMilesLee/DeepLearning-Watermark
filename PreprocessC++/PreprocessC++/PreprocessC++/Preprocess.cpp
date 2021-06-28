#include "Preprocess.h"

void Preprocess::ReadImageFromFile()
{
  cv::String file_location = "D:\\CS-Related\\Watermark Faker\\Test_Images\\facades\\test";
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
  std::cout << "Done. " << std::endl;
}

void Preprocess::CropImageLeftSide()
{
  std::cout << "The size of the vector is: " << image_Readin.size() << std::endl;
  cv::Rect CropArea(0, 0, 256, 256); //²ÃÇÐÎª256ÏñËØµÄÍ¼Æ¬
  for (size_t looptimes = 0; looptimes < filenames.size(); looptimes++)
  {
    cv::Mat imageCropTool = cv::Mat(image_Readin.at(looptimes), CropArea);
    image_Cropped.push_back(imageCropTool);
  }
  std::cout << std::endl << ".....Done. " << std::endl;
}

void Preprocess::CropImageRightSide()
{
  std::cout << "The size of the vector is: " << image_Readin.size() << std::endl;
  cv::Rect CropArea(256, 0, 256, 256); //²ÃÇÐÎª256ÏñËØµÄÍ¼Æ¬
  for (size_t looptimes = 0; looptimes < filenames.size(); looptimes++)
  {
    cv::Mat imageCropTool = cv::Mat(image_Readin.at(looptimes), CropArea);
    image_Cropped.push_back(imageCropTool);
  }
  std::cout << std::endl << ".....Done. " << std::endl;
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