import cv2
import random
import numpy as np

# 需要原图提取水印
image_path = r'E:\MyWork_aboutWatermark\fre_watermark\5g.bmp'
wmed_path = r'E:\MyWork_aboutWatermark\fre_watermark\5g-new-watermarked.bmp'

image = cv2.imread(image_path, 0)
wmed = cv2.imread(wmed_path, 0)

image_f = np.fft.fft2(image, axes=(0, 1))
wmed_f = np.fft.fft2(wmed, axes=(0, 1))

alpha, i_height, i_width = 5, image.shape[0], image.shape[1]

result = np.real((wmed_f - image_f) / alpha)

cv2.imwrite(r'E:\MyWork_aboutWatermark\fre_watermark\5g-extract-new.png', result)

print("finished")