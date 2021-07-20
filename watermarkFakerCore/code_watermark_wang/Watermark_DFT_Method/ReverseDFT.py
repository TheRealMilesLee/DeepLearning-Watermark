import numpy as np
import cv2
import os
import glob

def de_fourier_watermark(OriginalImage, WatermarkedImage):
    OriginalImage_Fourier = np.fft.fft2(OriginalImage, axes=(0, 1))
    WatermarkedImage_Fourier = np.fft.fft2(WatermarkedImage, axes=(0, 1))
    WatermarkContrast = 15
    result = np.real((WatermarkedImage_Fourier - OriginalImage_Fourier) / WatermarkContrast)
    return result

original_path = "D:/CS-Related/Watermark Faker/TestFolder/Test_Images/DFT/Train images/Source"  # 与水印图片配对的原始图片
watermarked_path = "D:/CS-Related/Watermark Faker/TestFolder/Test_Images/DFT/Train images/Output"  # 加了水印的图片
output_path = "D:/CS-Related/Watermark Faker/TestFolder/Test_Images/DFT/Train images/Output/Reverse"  # 输出提取的水印的地方

# 读取与水印匹配的源文件
Source_Filename = glob.glob(os.path.join(original_path, "*.bmp"))

# 读取带水印的图片文件
Watermark_Filename = glob.glob(os.path.join(watermarked_path, "*.png"))

#进行反向水印提取
for order in range(len(Source_Filename)):
    # Exception Handling
    if not os.path.exists(watermarked_path):
        print("配对的水印图片不存在")
    
    #读入与水印配对的源文件
    Source = cv2.imread(Source_Filename[order], 1)  
    
    # 读入带水印的文件
    Watermark = cv2.imread(Watermark_Filename[order], 1)
    
    # Core fourier transfer
    de_watermark = de_fourier_watermark(Source, Watermark)
    
    #Add more lighting to the watermark
    res1 = np.uint8(np.clip((cv2.add(1.8*de_watermark,30)), 0, 255))

    #Preparing for output
    name = os.path.basename(Source_Filename[order]).split(".")[0]
    save_path = os.path.join(output_path, name + "_de.png")
    print(save_path)
    
    cv2.imwrite(save_path, res1)
