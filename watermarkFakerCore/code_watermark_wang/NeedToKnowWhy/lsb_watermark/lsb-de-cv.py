import cv2 as cv
import numpy as np
import os
import glob
# 文件目录位置
input_path = '/Users/arkia/ComputerScienceRelated/Watermark Faker/Watermark Faker Data/LSB/output'
output_path = '/Users/arkia/ComputerScienceRelated/Watermark Faker/Watermark Faker Data/LSB/output/Reverse'
output_extract = glob.glob(os.path.join(input_path, "*-outputs.png"))
target_extract = glob.glob(os.path.join(input_path, "*-targets.png"))
# 错误处理
if len(output_extract) != len(target_extract):
    raise RuntimeError
if not os.path.exists(output_path):
    os.makedirs(output_path)

"""
Removes all the significant bits and place the least significant bits 
to the most significant for watermark extraction.
"""
def dewatermark_lsb(result):
    result = format(result, '08b')
    info = result[-2:]
    wm = int(info + '0'*6, 2)
    return wm

for order in range(len(output_extract)):
    print(order + 1)
    # 文件读入
    output = cv.imread(output_extract[order])
    target = cv.imread(target_extract[order])
    #反向转换提取水印
    vec_dewatermark = np.vectorize(dewatermark_lsb)  
    # 输出准备
    extracted_out = vec_dewatermark(output)
    extracted_tar = vec_dewatermark(target)
     # 文件重命名
    out_rename = os.path.splitext(os.path.basename(output_extract[order]))[0] + "-extract.png"
    tar_rename = os.path.splitext(os.path.basename(target_extract[order]))[0] + "-extract.png"
    # 输出
    cv.imwrite(os.path.join(output_path, out_rename), extracted_out)
    cv.imwrite(os.path.join(output_path, tar_rename), extracted_tar)
print("finish")