import cv2 as cv
import numpy as np
import os
import glob

#文件目录
input_path = '/Users/arkia/ComputerScienceRelated/watermarkFakerCore/code_watermark_wang/lsb_watermark/input/samples_0.jpg'
output_path = '/Users/arkia/ComputerScienceRelated/watermarkFakerCore/code_watermark_wang/lsb_watermark/output'
watermark = cv.imread('/Users/arkia/ComputerScienceRelated/watermarkFakerCore/code_watermark_wang/dft_watermark/lena_color_256.png')
input_extract = glob.glob(os.path.join(input_path, "*.png"))

"""
Watermarks the image by replacing the least significant bits of the image.
"""
def watermark_lsb(origin, info):
    origin = format(origin, '08b')  # 8 bits binary number in str filled by '0' (255 = 2**8)
    info = format(info, '08b')
    wm_msb = info[:2]  # the most significant bits of info
    im_lsb = origin[:-2]  # remove the least significant bits of origin
    result = int(im_lsb + wm_msb, 2)  # turn to decimal integer
    return result

#如果输出目录不存在就创建一个
if not os.path.exists(output_path):
    os.mkdir(output_path)

for order in range(len(input_extract)):
#输出处理的储量
    print(order + 1)
    #按照之前input_extract存储的文件名来读取原文件夹中的文件
    origin = cv.imread(input_extract[order])
    # Make watermark image's size as original image，也就是铺垫一圈0以防后续操作损坏原图片
    wmbroad = np.zeros(origin.shape)
    
    wmbroad[:watermark.shape[0], :watermark.shape[1], :] = watermark
    wmbroad = wmbroad.astype('uint8')  # 8 bits binary numbers
    # To embed watermarking
    vec_watermark = np.vectorize(watermark_lsb)  # to apply LSB to matrices and arrays
    watermarked = vec_watermark(origin, wmbroad)
    input_rename = os.path.splitext(os.path.basename(input_extract[order]))[0] + "_wLSB.png"
#输出所有东西
    cv.imwrite(os.path.join(output_path, input_rename), watermarked)

print("finish")
