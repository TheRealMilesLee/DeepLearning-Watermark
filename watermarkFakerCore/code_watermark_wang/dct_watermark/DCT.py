import os
import cv2
import argparse
import numpy as np
from watermarks import dct

if __name__ == '__main__':
    alg = dct.DCT()
    source_path = "/Users/arkia/Pictures/The-Wallpaper-Collection/Year of 2021"
    watermark_path = "/Users/arkia/ComputerScienceRelated/Watermark Faker/DeepLearning-WatermarkFaker/watermarkFakerCore/code_watermark_wang/dct_watermark/images/lena.png"
    output_path = "/Users/arkia/ComputerScienceRelated/Watermark Faker/Watermark Faker Data/Output"
    RGB = 0

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    i = 0
    for file_name in os.listdir(source_path):
        image_path = os.path.join(source_path, file_name)
        tmp_str = file_name.split(".")[0]
        image_wm_path = os.path.join(output_path, tmp_str + "_wDCT.png")

        image = cv2.imread(image_path, flags=RGB)
        watermark = cv2.imread(watermark_path, flags=RGB)

        watermarked_image = alg.embed(image, watermark)

        cv2.imwrite(image_wm_path, watermarked_image)

        i += 1
        print(i)


    print("finished")