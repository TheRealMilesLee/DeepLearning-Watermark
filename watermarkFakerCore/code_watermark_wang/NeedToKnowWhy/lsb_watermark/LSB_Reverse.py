import cv2 as cv
import numpy as np
import os
import glob

# 文件目录位置
input_path = '/Users/arkia/ComputerScienceRelated/Watermark Faker/Watermark Faker Data/LSB/output'
output_path = '/Users/arkia/ComputerScienceRelated/Watermark Faker/Watermark Faker Data/LSB/output/Reverse'
output_extract = glob.glob(os.path.join(input_path, "*.png"))
# 错误处理
if not os.path.exists(output_path):
    os.makedirs(output_path)

"""
Removes all the significant bits and place the least significant bits 
to the most significant for watermark extraction.
"""
def dewatermark_lsb(result):
    # 转化为二进制模式
    result = format(result, '08b')
    info = result[-2:]
    wm = int(info + '0'*6, 2)
    return wm

for order in range(len(output_extract)):
    print(order + 1)
    # 文件读入
    output = cv.imread(output_extract[order])
    #反向转换提取水印
    vec_dewatermark = np.vectorize(dewatermark_lsb)  
    # 输出准备
    extracted_out = vec_dewatermark(output)
     # 文件重命名
    out_rename = os.path.splitext(os.path.basename(output_extract[order]))[0] + "-extract.png"
    # 输出
    cv.imwrite(os.path.join(output_path, out_rename), extracted_out)
print("finish")