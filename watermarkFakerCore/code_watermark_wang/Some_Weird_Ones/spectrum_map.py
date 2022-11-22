import cv2 as cv
import os
from tqdm import tqdm
import numpy as np

# 该代码用于将图转换到DCT域，然后转换为对数频谱图

input_dir = "/home/wangruowei/PycharmProjects/watermark2020/experment_results/dct/test_result_pix2pix/images"
output_dir = "/home/wangruowei/PycharmProjects/watermark2020/experment_results/dct/test_result_pix2pix/images_fre_no20"

# 如果输出文件夹不存在，就创建它
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# 对于input_dir中的每一个图像，将其读取出来，进行dct变换，保存为频谱图
for image_name in tqdm(os.listdir(input_dir)):
    image = cv.imread(os.path.join(input_dir, image_name), flags=0)
    image = image.astype(np.float32)
    image = cv.dct(image)
    image = 20*np.log(image)
    # image = np.log(image)

    name = image_name.split(".")[0]
    cv.imwrite(os.path.join(output_dir, name + "_log.png"), image)

print("finished")

