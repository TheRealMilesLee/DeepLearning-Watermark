import numpy as np
import cv2 as cv
from skimage import measure

# 该代码用于测试傅里叶变换相关

image = cv.imread(r"E:\MyWork_aboutWatermark\fre_watermark\DUTS-TR-Image_4212_gray.png", flags=0)
watermark = cv.imread(r"E:\MyWork_aboutWatermark\fre_watermark\lena_grey_128.png", flags=0)

image_dft = np.fft.fft2(image, axes=(0, 1))
# # image_dft = np.fft.fftshift(image_dft, axes=(0, 1))  # 该函数会对调4个象限，对调方式参见https://zhuanlan.zhihu.com/p/99605178?utm_source=qq
image_dft_log = np.log(1 + abs(image_dft))
image_dft_log = image_dft_log * 256 / (image_dft_log.max() - image_dft_log.min())

cv.imwrite(r"E:\MyWork_aboutWatermark\fre_watermark\DUTS-TR-Image_4212_gray-fre.png", abs(image_dft))
cv.imwrite(r"E:\MyWork_aboutWatermark\fre_watermark\DUTS-TR-Image_4212_gray-frelog.png", image_dft_log)


# x = measure.compare_mse(image, image)
# print(x)
# print("finished")

# i_height, i_width = image.shape[0], image.shape[1]
# w_height, w_width = watermark.shape[0], watermark.shape[1]
#
# print(image)
# print(image.shape)
#
# image_dft = np.fft.fft2(watermark, axes=(0, 1))
# # # image_dft = np.fft.fftshift(image_dft, axes=(0, 1))  # 该函数会对调4个象限，对调方式参见https://zhuanlan.zhihu.com/p/99605178?utm_source=qq
# image_dft = np.log(1 + abs(image_dft))
#
# image_dft = image_dft * 255 / np.max(image_dft) - np.min(image_dft)
#
# print(image_dft)
#
# # cv.namedWindow('input_image', cv.WINDOW_AUTOSIZE)
# # cv.imshow('input_image', image_dft)
# # cv.waitKey(0)
# # cv.destroyAllWindows()
#
# cv.imwrite(r"E:\MyWork_aboutWatermark\fre_watermark\DUTS-TR-Image_4212_fft.png", image_dft)
#
# print("finished")
