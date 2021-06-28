import numpy as np
import cv2
import os
import glob


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
watermarked_path = r"F:\image_database\grey_watermark_frequency-backup"  # 加了水印的图片
output_path = r"F:\image_database\gery_dewatermark_test"  # 输出提取的水印的地方

originals = glob.glob(os.path.join(original_path, "*.png"))
watermarkeds = glob.glob(os.path.join(watermarked_path, "*.png"))

for order in range(len(originals)):
    print(order)

    basename, _ = os.path.splitext(os.path.basename(originals[order]))
    watermark_name = basename + "_freWatermark"

    sibling_path = os.path.join(watermarked_path, basename + "_freWatermark" + ".png")

    if not os.path.exists(sibling_path):
        print("配对的水印图片不存在")

    image = cv2.imread(originals[order], 0)  # 后面加入了0之后，读取的就是单通道的灰度图了
    wmed = cv2.imread(sibling_path, 0)

    de_watermark = de_fourier_watermark(image, wmed)

    name = os.path.basename(originals[order]).split(".")[0]
    save_path = os.path.join(output_path, name + "_de.png")
    print(save_path)

    cv2.imwrite(save_path, de_watermark)