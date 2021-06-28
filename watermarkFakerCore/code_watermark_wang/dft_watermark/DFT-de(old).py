import cv2
import random
import numpy as np

# 需要原图提取水印
image_path = r'E:\MyWork_aboutWatermark\fre_watermark\5.bmp'
wmed_path = r'E:\MyWork_aboutWatermark\fre_watermark\watered.png'

image = cv2.imread(image_path)
wmed = cv2.imread(wmed_path)

image_f = np.fft.fft2(image)
wmed_f = np.fft.fft2(wmed)

alpha, i_height, i_width = 1, image.shape[0], image.shape[1]

# 乱序水印
shuffled_wm = np.real((wmed_f - image_f) / alpha)

# 使用嵌入时相同的seed
random.seed(i_height + i_width)
x, y = list(range(int(i_height / 2))), list(range(i_width))
random.shuffle(x)
random.shuffle(y)

# 重组乱序的水印得到原水印
result = np.zeros(shuffled_wm.shape)
for i in range(int(i_height / 2)):
    for j in range(i_width):
        result[x[i]][y[j]] = shuffled_wm[i][j]

cv2.imwrite(r'E:\MyWork_aboutWatermark\fre_watermark\extract.png', result)