#include "Preprocess.h"

int main()
{
  try
  {
    std::string userArgumentMode;
    std::cout << "Which Crop Direction: ";
    std::cin >> userArgumentMode;
    Preprocess preprocessObject;
    if (userArgumentMode == "AtoB")
    {
      std::cout << std::endl << "Readling images from the desk...";
      preprocessObject.ReadImageFromFile();
      std::cout << std::endl << "Starting crop the image to 256x256...";
      preprocessObject.CropImageLeftSide();
      std::cout << std::endl << "Starting convert image into [-1,1]";
      preprocessObject.ConvergentImage();
      std::cout << std::endl << "Output everything to the csv file...";
      preprocessObject.OutputAsCSV();
      std::cout << std::endl << "Done.";
    }
    else if (userArgumentMode == "BtoA")
    {
      std::cout << std::endl << "Readling images from the desk...";
      preprocessObject.ReadImageFromFile();
      std::cout << std::endl << "Starting crop the image to 256x256...";
      preprocessObject.CropImageRightSide();
      std::cout << std::endl << "Starting convert image into [-1,1]";
      preprocessObject.ConvergentImage();
      std::cout << std::endl << "Output everything to the csv file...";
      preprocessObject.OutputAsCSV();
      std::cout << std::endl << "Done.";
    }
    else
    {
      throw StorageException();
    }
  } catch (const char *error)
  {
    std::cout << "Error: " << error << std::endl;
  }

  return 0;
}