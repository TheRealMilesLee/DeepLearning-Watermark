import os
import cv2
import argparse
import numpy as np
from watermarks import lsbm

parser = argparse.ArgumentParser()
parser.add_argument("--source_path",
                    default="/home/wangruowei/PycharmProjects/watermark2020/experment_results/dct/test_newDCT_noTanh_bn(l1_weight=100)/images")
parser.add_argument("--output_path",
                    default="/home/wangruowei/PycharmProjects/watermark2020/experment_results/dct/extract_newDCT_noTanh_bn(l1_weight=100)")
a = parser.parse_args()

if __name__ == '__main__':
    print("start")
    algorithm = lsbm.DCT
    source_path = a.source_path  # watermarked images
    output_path = a.output_path  # extracted watermark
    RGB = 0

    alg = algorithm()
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    i = 0
    for file_name in os.listdir(source_path):
        # if file_name.__contains__("input"):
        #     continue

        image_wm_path = os.path.join(source_path, file_name)
        origin_name = file_name.split("-")[0] + ".png"

        image_wm = cv2.imread(image_wm_path)

        extrated_watermark_path = os.path.join(output_path, file_name.split(".")[0] + "-extract.png")
        extrated_watermark = alg.extract(image_wm)
        cv2.imwrite(extrated_watermark_path, extrated_watermark)

        i += 1
        print(i)


    print("finished")