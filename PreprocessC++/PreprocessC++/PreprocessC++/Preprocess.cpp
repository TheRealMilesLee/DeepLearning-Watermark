#include "Preprocess.h"

void Preprocess::ReadImageFromFile()
{
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
  std::cout << std::endl << "The size of the vector is: " << image_Readin.size() << std::endl;
  for (size_t loop = 0; loop < image_Cropped.size(); loop++)
  {
    cv::Mat NormalConvertImage = image_Cropped.at(loop) / 255;
    cv::Mat ConvertedImage = NormalConvertImage / 2 - 1;
    NormalizedImage.push_back(ConvertedImage);
  }
  std::cout << std::endl << ".....Done. " << std::endl;
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