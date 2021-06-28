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
      preprocessObject.ReadImageFromFile();
      preprocessObject.CropImageLeftSide();
      preprocessObject.ConvergentImage();
      preprocessObject.OutputAsCSV();
    }
    else if (userArgumentMode == "BtoA")
    {
      preprocessObject.ReadImageFromFile();
      preprocessObject.CropImageRightSide();
      preprocessObject.ConvergentImage();
      preprocessObject.OutputAsCSV();
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