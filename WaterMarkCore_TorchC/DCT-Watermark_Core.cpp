#include "DCT_Watermark.h"

int main()
{
  DCT_Watermark watermarkObject;
  watermarkObject.ReadImageFromDisk();
  watermarkObject.Generator();
  /*
  watermarkObject.GetCSVFromDisk();
  watermarkObject.CoreTensorModule();
  watermarkObject.CheckpointOutput();
   */
  return 0;
}
