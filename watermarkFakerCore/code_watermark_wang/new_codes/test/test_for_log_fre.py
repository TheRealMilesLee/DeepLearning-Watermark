import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


'''↓，测试，tf.log()'''
'''结论：'''
# x = tf.constant([-8, -1, 0, 1, 8], dtype=tf.float32)
# y = tf.log(x)
#
# with tf.Session() as sess:
#     z = y.eval()



'''↓，测试，对负数，求取-log(-x+1)；对正数，求取log(x+1)'''
'''结论：成功完成了归一化，但该方法下，差别变小了'''


def new_log(x):
    if x >= 0:
        y = np.log(x + 1)
    else:
        y = -np.log(-x + 1)
    return y


def new_log2(x):
    if x >= 0:
        y = np.log(x)
    else:
        y = -np.log(-x)
    return y


image_path = "/home/wangruowei/PycharmProjects/watermark2020/code_watermark/dct_watermark/images/lena.png"

image = cv.imread(image_path, flags=0)
image = image.astype(np.float32)

image_1 = image / 255

image_dct1 = cv.dct(image_1)
image_dct1_processsed = np.reshape(np.array(list(map(lambda x: new_log(x), np.nditer(image_dct1)))), [256, 256])
image_dct1_processsed2 = np.reshape(np.array(list(map(lambda x: new_log2(x), np.nditer(image_dct1)))), [256, 256])

mean = np.mean(image_dct1_processsed)
var = np.var(image_dct1_processsed)
image_dct1_processsed = (image_dct1_processsed - mean) / np.sqrt(var)

# # 下面这种情况是为了讲正负切分而制作的
# log_image_dct1 = np.log(np.reshape(np.array(list(map(lambda x: new_log2(x), np.nditer(image_dct1)))), [256, 256]))

plt.subplot(121), plt.imshow(image_dct1_processsed, "gray"), plt.title("image_dct1_processsed")
plt.subplot(122), plt.imshow(image_dct1_processsed2, "gray"), plt.title("image_dct1_processsed2")

'''↓，测试，求取图像的均值和方差'''
'''结论：进行z-sore标准化后，变成了正太分布，均值为0，最大值为2.4，最小值为-2.0、'''
# image_path = "/home/wangruowei/PycharmProjects/watermark2020/code_watermark/dct_watermark/images/lena.png"
#
# image = cv.imread(image_path, flags=0)
# image = image.astype(np.float32)
#
# mean = np.mean(image)  # 求均值
# var = np.var(image)  # 求方差
#
# EPS = 1e-12
# image_norm = (image - mean) / np.sqrt(var + EPS)

'''↓，测试，将dct()后的值，加512，再log()后结果怎样。'''
'''结论：进行上述操作后，图像完全黑掉了，从值上来看，log()后的值，差距很小'''
# image_path = "/home/wangruowei/PycharmProjects/watermark2020/code_watermark/dct_watermark/images/lena.png"
#
# image = cv.imread(image_path, flags=0)
# image = image.astype(np.float32)
#
# image_1 = image / 255
#
# image_dct1 = cv.dct(image_1)
# image_dct1_plus = image_dct1 + 512
#
# log_image_dct1 = np.log(np.abs(image_dct1))
# log_image_dct1_plus = np.log(image_dct1_plus)
#
# plt.subplot(121), plt.imshow(log_image_dct1, "gray"), plt.title("log_image_dct1")
# plt.subplot(122), plt.imshow(log_image_dct1_plus, "gray"), plt.title("log_image_dct255")

'''↓，测试log()运算，以及还原'''
'''结论：使用以下操作后，还原正常'''
# x0 = 10
# x1 = np.log(x0)
# x2 = np.exp(x1)

'''↓，如果将dct()运算后的值取绝对值，然后再转换回去会怎样。'''
'''结论：会出现黑屏现象'''
# image_path = "/home/wangruowei/PycharmProjects/watermark2020/code_watermark/dct_watermark/images/lena.png"
#
# image = cv.imread(image_path, flags=0)
# image = image.astype(np.float32)
#
# image_255 = image
#
# image_dct255 = cv.dct(image_255)
# image_dct255_abs = abs(image_dct255)
#
# image_idct255 = cv.idct(image_dct255)
# image_idct255_abs = cv.idct(image_dct255_abs)
#
# plt.subplot(121), plt.imshow(image_idct255, "gray"), plt.title("image_idct255")
# plt.subplot(122), plt.imshow(image_idct255_abs, "gray"), plt.title("image_idct255_abs")

'''↓，测试log运算后得到的结果怎样。'''
'''结论：数据的动态范围变小，而且分布良好'''
# image_path = "/home/wangruowei/PycharmProjects/watermark2020/code_watermark/dct_watermark/images/lena.png"
#
# image = cv.imread(image_path, flags=0)
# image = image.astype(np.float32)
#
# image_1 = image / 255
# image_255 = image
#
# image_dct1 = cv.dct(image_1)
# image_dct255 = cv.dct(image_255)
#
# log_image_dct1 = np.log(np.abs(image_dct1))
# log_image_dct255 = np.log(np.abs(image_dct255))
#
# plt.subplot(121), plt.imshow(log_image_dct1, "gray"), plt.title("log_image_dct1")
# plt.subplot(122), plt.imshow(log_image_dct255, "gray"), plt.title("log_image_dct255")
