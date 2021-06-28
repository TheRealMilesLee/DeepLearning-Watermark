import cv2
import random
import numpy as np


# 对于彩图，应该对每个色道分别做傅里叶变换，再嵌入水印
# 而np.fft.fft2(color_image)中参数axes默认为(-2, -1)，是对width、channel两维度做傅里叶变换，无意义
# 故axes应为(0, 1)，处理height、width两维

# DFT对彩色图片嵌入彩色水印的效果不好，水印图像大致正确，但颜色偏离较为严重
# 所以建议用DFT处理灰度图像与灰度水印
# 当然也可以将灰度水印嵌入到彩色图像的某一色道（e.g. 人眼不敏感的B色道）
# 为训练神经网络方便，建议对于DFT水印采用灰度图片，这样网络训练的速度也更快
# （同样，考虑到神经网络学习能力，此处DFT算法没有对嵌入水印置乱）


# 读取图片信息
# 水印图片的大小要小于要加入水印图片大小的一半(为了成中心对称)
image_path = r'E:\MyWork_aboutWatermark\fre_watermark\5g.bmp'
watermark_path = r'E:\MyWork_aboutWatermark\fre_watermark\lena_grey_256.png'

image = cv2.imread(image_path, 0)  # 灰度图片
watermark = cv2.imread(watermark_path, 0)  # 灰度水印
i_height, i_width = image.shape[0], image.shape[1]
w_height, w_width = watermark.shape[0], watermark.shape[1]

# 傅里叶变换
image_f = np.fft.fft2(image, axes=(0, 1))  # 若输入为灰度图片，可使用默认axes；若彩色，需指定axes=(0, 1)

tmp = np.zeros(image.shape)

# 图片经过傅里叶变换后，水印图片直接按像素覆盖到频率域，因为频谱是中心对称的，
# 所以加水印也要对称的加，具体就是图片分上下两部分，左上加了什么，右下也要加同样的内容
for i in range(w_height):
    for j in range(w_width):
        tmp[i][j] = watermark[i][j]
        tmp[i_height - i - 1][i_width - 1 - j] = tmp[i][j]
# cv2.imwrite(r'board_wm.png', tmp)

# 调试alpha后发现规律：alpha越大图片越清晰，但亮度越暗，反之则相反
alpha = 5  # 水印图片加进去的强度(混合的比例因子)
result_f = image_f + alpha * tmp

# 反傅里叶变换i_height, i_width, i_channel = np.shape(image)
# w_height, w_width, w_channel = np.shape(watermark)
result = np.real(np.fft.ifft2(result_f, axes=(0, 1)))

# 保存加了水印后的图片
cv2.imwrite(r'E:\MyWork_aboutWatermark\fre_watermark\5g-new-watermarked.bmp', result)

print("finished")
