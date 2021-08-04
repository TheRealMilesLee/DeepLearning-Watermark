import os
import cv2
from watermarkAlgorithmLibrary import dct

if __name__ == '__main__':
  # 源文件位置
  source_path = "D:/CS-Related/Watermark Faker/TestFolder/Test_Images/DCT/images/Source"
  watermark_path = "D:/CS-Related/Watermark Faker/TestFolder/Test_Images/DCT/images/Watermark/lena.png"
  output_path = "D:/CS-Related/Watermark Faker/TestFolder/Test_Images/DCT/images/Output"
  # 统一设置为灰度值
  RGB = 0
  # 如果输出的位置文件夹不存在就创建一个新的
  if not os.path.exists(output_path):
    os.mkdir(output_path)
    
  i = 0
  for file_name in os.listdir(source_path):
    image_path = os.path.join(source_path, file_name)
    tmp_str = file_name.split(".")[0]
    image_wm_path = os.path.join(output_path, tmp_str + "_wDCT.png")
    
    image = cv2.imread(image_path, flags = RGB)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    
    watermarkLena = cv2.imread(watermark_path, flags = RGB)
    cv2.imshow('Watermark', watermarkLena)
    cv2.waitKey(0)
    
    ResizeWatermarkHeight = watermarkLena.shape[0] / 8
    ResizeWatermarkWidth = watermarkLena.shape[1] / 8
    NewWatermarkImage = (ResizeWatermarkWidth, ResizeWatermarkHeight)
    
    resized = cv2.resize(watermarkLena, None, fx = 0.125, fy = 0.125, interpolation = cv2.INTER_AREA)
    
    watermarked_image = alg.embed(image, resized)
    
    cv2.imwrite(image_wm_path, watermarked_image)
    
    i += 1
    print(i)
  
  print("finished")
