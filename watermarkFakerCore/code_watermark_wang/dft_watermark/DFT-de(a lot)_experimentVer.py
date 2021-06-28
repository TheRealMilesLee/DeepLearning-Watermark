import numpy as np
import cv2
import os
import glob

# 2020年4月15日15点05分，该代码版本是更方便处理模型的输出而创建的。基于DFT-de(a lot).py

def de_fourier_watermark(image, wmed):
    image_f = np.fft.fft2(image, axes=(0, 1))
    wmed_f = np.fft.fft2(wmed, axes=(0, 1))

    alpha, i_height, i_width = 5, image.shape[0], image.shape[1]

    result = np.real((wmed_f - image_f) / alpha)

    return result

# --------------原始设置--------------------
# original_path = r"F:\image_database\grey"
# watermarked_path = r"F:\image_database\grey_watermark_frequency"
# output_path = r"F:\image_database\gery_dewatermark_test"

original_path = r"F:\image_database\grey"  # 与水印图片配对的原始图片
watermarked_path = r"E:\freWatermark_experiment\experiment2_weishu\test_result(2020.4.15_0.07942)\images"  # 加了水印的图片
output_path = r"E:\freWatermark_experiment\experiment2_weishu\extract_result(2020.4.15_0.07942)"  # 输出提取的水印的地方

if not os.path.exists(output_path):
    os.makedirs(output_path)

outputs = glob.glob(os.path.join(watermarked_path, "*-outputs.png"))
targets = glob.glob(os.path.join(watermarked_path, "*-targets.png"))

for order in range(len(outputs)):
    print(order)

    basename, _ = os.path.splitext(os.path.basename(outputs[order]))
    origin_name = basename[0:-8]  # 用于train文件夹中的训练过程中产生的图片处理，设置[9:-8]，用于test文件夹中的图片处理，设置[0:-8]

    basename2, _ = os.path.splitext(os.path.basename(targets[order]))
    target_name = basename2[0:-8]  # 用于train文件夹中的训练过程中产生的图片处理，设置[9:-8]，用于test文件夹中的图片处理，设置[0:-8]

    sibling_path = os.path.join(original_path, origin_name + ".png")

    if not os.path.exists(sibling_path):
        print("配对的水印图片不存在")
        break

    if not origin_name == target_name:
        print("目标图与输出图不匹配")
        break

    image = cv2.imread(sibling_path, 0)  # 后面加入了0之后，读取的就是单通道的灰度图了
    output_watermarked = cv2.imread(outputs[order], 0)
    target_watermarked = cv2.imread(targets[order], 0)

    de_watermark_out = de_fourier_watermark(image, output_watermarked)
    de_watermark_tar = de_fourier_watermark(image, target_watermarked)

    save_path_out = os.path.join(output_path, basename + "_de.png")
    print(save_path_out)
    cv2.imwrite(save_path_out, de_watermark_out)

    save_path_tar = os.path.join(output_path, basename2 + "_de.png")
    print(save_path_tar)
    cv2.imwrite(save_path_tar, de_watermark_tar)

print("finished")