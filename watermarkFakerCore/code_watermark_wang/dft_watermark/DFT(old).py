import cv2
import random
import numpy as np

# 读取图片信息
# 水印图片的大小要小于要加入水印图片大小的一半(为了成中心对称)
image_path = r'E:\MyWork_aboutWatermark\fre_watermark\5.bmp'
watermark_path = r'E:\MyWork_aboutWatermark\fre_watermark\lena_color_256.png'

image = cv2.imread(image_path)
watermark = cv2.imread(watermark_path)

i_height, i_width, i_channel = np.shape(image)
w_height, w_width, w_channel = np.shape(watermark)

# 傅里叶变换
image_f = np.fft.fft2(image)

# 加水印的时候可以把频谱或水印图片的位置随机化，但要记住这个位置信息
# 用某个固定的随机数种子，解水印的时候恢复到随机位置
x, y = list(range(int(i_height / 2))), list(range(i_width))
random.seed(i_height + i_width)
random.shuffle(x)
random.shuffle(y)

# 图片经过傅里叶变换后，水印图片直接按像素覆盖到频率域，因为频谱是中心对称的，
# 所以加水印也要对称的加，具体就是图片分上下两部分，左上加了什么，右下也要加同样的内容
tmp = np.zeros(image.shape)
for i in range(int(i_height / 2)):
    for j in range(i_width):
        if x[i] < w_height and y[j] < w_width:
            tmp[i][j] = watermark[x[i]][y[j]]
            tmp[i_height - i - 1][i_width - 1 - j] = tmp[i][j]

alpha = 1  # 水印图片加进去的强度(混合的比例因子)
result_f = image_f + alpha * tmp

# 反傅里叶变换
result = np.real(np.fft.ifft2(result_f))

# 保存加了水印后的图片
cv2.imwrite(r'E:\MyWork_aboutWatermark\fre_watermark\watered.png', result)