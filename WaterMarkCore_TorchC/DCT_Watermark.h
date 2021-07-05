
#ifndef WATERMARKCOREC___DCT_WATERMARK_H
#define WATERMARKCOREC___DCT_WATERMARK_H

#include<iostream>
#include <torch/torch.h>
#include <vector>

class DCT_Watermark
{
private:
  std::vector<double> preprocessReadIn;
public:
  void Generator();
  void GetCSVFromDisk();
  void CoreTensorModule();
  void CheckpointOutput();
  
};


#endif //WATERMARKCOREC___DCT_WATERMARK_H
