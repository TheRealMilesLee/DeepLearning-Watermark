import numpy as np
import cv2
import os
import glob

original_path = "D:/CS-Related/Watermark Faker/TestFolder/Test_Images/DFT/Train images/Source"  # 与水印图片配对的原始图片
watermarked_path = "D:/CS-Related/Watermark Faker/TestFolder/Test_Images/DFT/Train images/Output/5_wDFT.png"  # 加了水印的图片
output_path = "D:/CS-Related/Watermark Faker/TestFolder/Test_Images/DFT/Train images/Output/Reverse"  # 输出提取的水印的地方

def de_fourier_watermark(OriginalImage, WatermarkedImage):
    OriginalImage_Fourier = np.fft.fft2(OriginalImage, axes=(0, 1))
    WatermarkedImage_Fourier = np.fft.fft2(WatermarkedImage, axes=(0, 1))
    alpha, i_height, i_width = 5, OriginalImage.shape[0], OriginalImage.shape[1]
    result = np.real((WatermarkedImage_Fourier - OriginalImage_Fourier) / alpha)
    return result

# 读取与水印匹配的源文件
Source_Filename = glob.glob(os.path.join(original_path, "*png"))
Source = cv2.imread(Source_Filename)
cv2.imshow('Source File', Source)
# 读取带水印的图片文件
Watermark_Filename = glob.glob(os.path.join(watermarked_path, "*.png"))
Watermark = cv2.imread(Watermark_Filename)
cv2.imshow('Watermarked Image', Watermark)
'''
for order in range(len(originals)):
    print(order)
    # 拆分文件名
    basename, _ = os.path.splitext(os.path.basename(originals[order]))
    # 为输出图片指定文件名
    watermark_name = basename + "_freWatermark"
    sibling_path = os.path.join(watermarked_path, basename + "_freWatermark" + ".png")
    if not os.path.exists(sibling_path):
        print("配对的水印图片不存在")
    NewImage = cv2.imread(originals[order], 0)  # 后面加入了0之后，读取的就是单通道的灰度图了
    Watermark = cv2.imread(sibling_path, 0)
    de_watermark = de_fourier_watermark(originals, Watermark)
    name = os.path.basename(originals[order]).split(".")[0]
    save_path = os.path.join(output_path, name + "_de.png")
    print(save_path)
    cv2.imwrite(save_path, de_watermark)

'''
