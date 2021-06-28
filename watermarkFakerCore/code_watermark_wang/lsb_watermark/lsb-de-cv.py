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


input_path = "/Users/arkia/ComputerScienceRelated/watermarkFakerCore/code_watermark_wang/lsb_watermark/input/samples_0.jpg"
output_path = "/Users/arkia/ComputerScienceRelated/watermarkFakerCore/code_watermark_wang/lsb_watermark/output"

output_extract = glob.glob(os.path.join(input_path, "*-outputs.png"))
target_extract = glob.glob(os.path.join(input_path, "*-targets.png"))

if len(output_extract) != len(target_extract):
    raise RuntimeError

if not os.path.exists(output_path):
    os.makedirs(output_path)

for order in range(len(output_extract)):
    print(order + 1)

    output = cv.imread(output_extract[order])
    target = cv.imread(target_extract[order])

    vec_dewatermark = np.vectorize(dewatermark_lsb)  # to extract LSB-watermarking
    extracted_out = vec_dewatermark(output)
    extracted_tar = vec_dewatermark(target)

    out_rename = os.path.splitext(os.path.basename(output_extract[order]))[0] + "-extract.png"
    tar_rename = os.path.splitext(os.path.basename(target_extract[order]))[0] + "-extract.png"

    cv.imwrite(os.path.join(output_path, out_rename), extracted_out)
    cv.imwrite(os.path.join(output_path, tar_rename), extracted_tar)

print("finish")