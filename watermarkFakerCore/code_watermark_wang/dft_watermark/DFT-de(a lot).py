import numpy as np
import cv2
import os
import glob

original_path = "D:/CS-Related/Watermark Faker/TestFolder/Test_Images/DFT/Train images/Source"  # 与水印图片配对的原始图片
watermarked_path = "D:/CS-Related/Watermark Faker/TestFolder/Test_Images/DFT/Train images/Output"  # 加了水印的图片
output_path = "D:/CS-Related/Watermark Faker/TestFolder/Test_Images/DFT/Train images/Output"  # 输出提取的水印的地方

'''
-------此处是傅里叶变换部分------
和之前的不同，这次是逆向变换，也就是从频域变换回时域
'''
def de_fourier_watermark(OriginalImage, WatermarkedImage):
    OriginalImage_Fourier = np.fft.fft2(OriginalImage, axes=(0, 1))
    WatermarkedImage_Fourier = np.fft.fft2(WatermarkedImage, axes=(0, 1))
    alpha, i_height, i_width = 5, OriginalImage.shape[0], OriginalImage.shape[1]
    result = np.real((WatermarkedImage_Fourier - OriginalImage_Fourier) / alpha)
    return result

# 读取与水印匹配的源文件
originals = glob.glob(os.path.join(original_path, "*.png"))
# 读取带水印的图片文件
Watermarked_Image = glob.glob(os.path.join(watermarked_path, "*.png"))

for order in range(len(originals)):
    print(order)
    # 拆分文件名
    basename, _ = os.path.splitext(os.path.basename(originals[order]))
    # 为输出图片指定文件名
    watermark_name = basename + "_freWatermark"
    sibling_path = os.path.join(watermarked_path, basename + "_freWatermark" + ".png")
    if not os.path.exists(sibling_path):
        print("配对的水印图片不存在")
    originals = cv2.imread(originals[order], 0)  # 后面加入了0之后，读取的就是单通道的灰度图了
    Watermarked_Image = cv2.imread(sibling_path, 0)
    de_watermark = de_fourier_watermark(originals, Watermarked_Image)
    name = os.path.basename(originals[order]).split(".")[0]
    save_path = os.path.join(output_path, name + "_de.png")
    print(save_path)
    cv2.imwrite(save_path, de_watermark)