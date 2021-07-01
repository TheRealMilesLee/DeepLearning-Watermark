#include "Preprocess.h"

void Preprocess::ReadImageFromFile()
{
  //Save the filename into the vector
  cv::glob(file_location, filenames);
  for (size_t looptimes = 0; looptimes < filenames.size(); looptimes++)
  {
    ImageReadIn = cv::imread(filenames.at(looptimes));
    image_Readin.push_back(ImageReadIn);
  }
  std::cout << "Done. " << std::endl;
}

void Preprocess::CropImageLeftSide()
{
  std::cout << "The size of the vector is: " << image_Readin.size() << std::endl;
  cv::Rect CropArea(0, 0, 256, 256); //²ÃÇÐÎª256ÏñËØµÄÍ¼Æ¬
  for (size_t looptimes = 0; looptimes < filenames.size(); looptimes++)
  {
    imageCropTool = cv::Mat(image_Readin.at(looptimes), CropArea);
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
    imageCropTool = cv::Mat(image_Readin.at(looptimes), CropArea);
    image_Cropped.push_back(imageCropTool);
  }
  std::cout << std::endl << ".....Done. " << std::endl;
}

void Preprocess::ConvergentImageAndOutput()
{
  std::cout << std::endl << "The size of the vector is: " << image_Readin.size() << std::endl;
  cv::Mat ConvertedImage;
  cv::Mat ToBeConverted;
  std::ofstream outfile;
  outfile.open("D:\\CS-Related\\Watermark Faker\\PreprocessDir\\Preprocess.csv");
  for (size_t loop = 0; loop < image_Cropped.size(); loop++)
  {
    ToBeConverted = image_Cropped.at(loop);
    ToBeConverted.convertTo(ConvertedImage, CV_32F, 2.0 / 255, -1);
    outfile << ConvertedImage;
  }
  outfile.close();
  std::cout << std::endl << ".....Done. " << std::endl;
}