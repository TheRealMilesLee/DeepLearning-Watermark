#pragma once
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <string>
#include <fstream>
#include <exception>

class StorageException : public std::runtime_error
{
public:
  StorageException() : std::runtime_error("Invalid Inputs") {}
};

class Preprocess
{
private:
  cv::String file_location = "D:\\CS-Related\\Watermark Faker\\Test_Images\\facades\\test";
  cv::Mat ImageReadIn;
  cv::Mat imageCropTool;
  std::vector<cv::String> filenames;
  std::vector<cv::Mat> image_Readin;
  std::vector<cv::Mat> image_Cropped;
public:
  void ReadImageFromFile();
  void CropImageLeftSide();
  void CropImageRightSide();
  void ConvergentImageAndOutput();
};