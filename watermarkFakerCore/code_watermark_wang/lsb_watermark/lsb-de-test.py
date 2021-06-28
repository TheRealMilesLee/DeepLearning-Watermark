import cv2 as cv
import numpy as np
import os
import glob


def dewatermark_lsb(result):
    """
    Removes all the significant bits and place the least significant bits to
    the most significant for watermark extraction.
    """
    result = format(result, '08b')
    info = result[-2:]
    wm = int(info + '0'*6, 2)
    return wm


input_path = "/home/wangruowei/PycharmProjects/watermark2020/data/watermark_lsb"
output_path = "/home/wangruowei/PycharmProjects/watermark2020/data/test"

output_extract = glob.glob(os.path.join(input_path, "*.png"))

if not os.path.exists(output_path):
    os.makedirs(output_path)

for order in range(len(output_extract)):
    print(order + 1)

    output = cv.imread(output_extract[order])

    vec_dewatermark = np.vectorize(dewatermark_lsb)  # to extract LSB-watermarking
    extracted_out = vec_dewatermark(output)

    out_rename = os.path.splitext(os.path.basename(output_extract[order]))[0] + "-extract.png"

    cv.imwrite(os.path.join(output_path, out_rename), extracted_out)

print("finish")