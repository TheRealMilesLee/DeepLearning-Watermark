import cv2 as cv
import numpy as np
import os
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_path",
                    default=None)
parser.add_argument("--output_path",
                    default=None)
a = parser.parse_args()

def dewatermark_lsb(result):
    """
    Removes all the significant bits and place the least significant bits to
    the most significant for watermark extraction.
    """
    result = format(result, '08b')
    info = result[-2:]
    wm = int(info + '0' * 6, 2)
    return wm


input_path = a.input_path
output_path = a.output_path

if not os.path.exists(output_path):
    os.mkdir(output_path)

input_list = glob.glob(os.path.join(input_path, "*.png"))

i = 0
for input_name in input_list:
    print(i)
    i = i + 1

    watered_image = cv.imread(input_name)

    vec_dewatermark = np.vectorize(dewatermark_lsb)  # to extract LSB-watermarking
    extracted = vec_dewatermark(watered_image)

    extracted_name = os.path.basename(input_name).split(".")[0] + "_extract.png"
    cv.imwrite(os.path.join(output_path, extracted_name), extracted)

print("finish")
