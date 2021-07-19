import numpy as np
import cv2
import os
import glob

def de_fourier_watermark(OriginalImage, WatermarkedImage):
    OriginalImage_Fourier = np.fft.fft2(OriginalImage, axes=(0, 1))
    WatermarkedImage_Fourier = np.fft.fft2(WatermarkedImage, axes=(0, 1))
    alpha, i_height, i_width = 5, OriginalImage.shape[0], OriginalImage.shape[1]
    result = np.real((WatermarkedImage_Fourier - OriginalImage_Fourier) / alpha)
    return result

original_path = "/Users/arkia/ComputerScienceRelated/Watermark Faker/Watermark Faker Data/DFT/Train images/Source"  # 与水印图片配对的原始图片
watermarked_path = "/Users/arkia/ComputerScienceRelated/Watermark Faker/Watermark Faker Data/DFT/Train images/Output"  # 加了水印的图片
output_path = "/Users/arkia/ComputerScienceRelated/Watermark Faker/Watermark Faker Data/DFT/Train images/Output/Reverse"  # 输出提取的水印的地方
# 读取与水印匹配的源文件
Source_Filename = glob.glob(os.path.join(original_path, "*.png"))
print(len(Source_Filename))
# 读取带水印的图片文件
Watermark_Filename = glob.glob(os.path.join(watermarked_path, "*.png"))
print(len(Watermark_Filename))
for order in range(len(Source_Filename)):
    print(order)
    if not os.path.exists(watermarked_path):
        print("配对的水印图片不存在")
    Source = cv2.imread(Source_Filename[order], 0)  # 后面加入了0之后，读取的就是单通道的灰度图了
    Watermark = cv2.imread(Watermark_Filename[order], 0)
    de_watermark = de_fourier_watermark(Source, Watermark)
    name = os.path.basename(Source_Filename[order]).split(".")[0]
    save_path = os.path.join(output_path, name + "_de.png")
    print(save_path)
    cv2.imwrite(save_path, de_watermark)
