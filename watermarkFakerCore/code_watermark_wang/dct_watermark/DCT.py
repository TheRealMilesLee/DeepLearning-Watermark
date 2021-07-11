import os
import cv2
from watermarkAlgorithmLibrary import dct

if __name__ == '__main__':
  alg = dct.DCT()
  source_path = "D:/CS-Related/Watermark Faker/TestFolder/Test_Images/DCT/images/Source/"
  watermark_path = "D:/CS-Related/Watermark Faker/TestFolder/Test_Images/DCT/images/Watermark/"
  output_path = "D:/CS-Related/Watermark Faker/TestFolder/Test_Images/DCT/images/Output/"
  RGB = 0
  
  if not os.path.exists(output_path):
    os.mkdir(output_path)
  
  i = 0
  for file_name in os.listdir(source_path):
    image_path = os.path.join(source_path, file_name)
    tmp_str = file_name.split(".")[0]
    image_wm_path = os.path.join(output_path, tmp_str + "_wDCT.png")
    
    image = cv2.imread(image_path, flags = RGB)
    watermark = cv2.imread(watermark_path, flags = RGB)
    
    watermarked_image = alg.embed(image, watermark)
    
    cv2.imwrite(image_wm_path, watermarked_image)
    
    i += 1
    print(i)
  
  print("finished")
