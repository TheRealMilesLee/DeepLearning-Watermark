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
  std::vector<cv::String> filenames;
  std::vector<cv::Mat> image_Readin;
  std::vector<cv::Mat> image_Cropped;
  std::vector<cv::Mat> NormalizedImage;
  std::vector<cv::Mat> OutputVector;

public:
  void ReadImageFromFile();
  void CropImageLeftSide();
  void CropImageRightSide();
  void ConvergentImage();
  void OutputAsCSV();
};