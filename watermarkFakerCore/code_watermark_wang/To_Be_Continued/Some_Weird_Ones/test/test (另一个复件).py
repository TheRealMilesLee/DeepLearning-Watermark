import cv2 as cv
import tensorflow as tf
import numpy as np
import os


# image_path = "/home/wangruowei/PycharmProjects/watermark2020/code_watermark/dft_watermark/lena_grey_256.png"
# # im_path = "/home/wangruowei/PycharmProjects/watermark2020/data/gray/test/007_0007.png"
# # im_dct_path = "/home/wangruowei/PycharmProjects/watermark2020/data/watermark_dct/007_0007_wDCT.png"
#
# image = cv.imread(image_path, flags=0)
# image = image.astype('float32') / 255
# x = cv.dct(image)
# x_2 = x / 10
# x_2[0, 0] = 1
# tmp = x_2 * 10
# y = cv.idct(tmp)
# y_save = y * 255
# cv.imwrite("y.png", y_save)
# # image = image.astype('float32')
# image = tf.image.decode_png()
# image_01 = image / 255

# image_tf = tf.read_file(filename=image_path)
# image_tf = tf.image.decode_png(image_tf, channels=0)
# image_tf = tf.image.convert_image_dtype(image_tf, dtype=tf.float32)
# #
# with tf.Session() as sess:
#     im = tf.read_file(filename=im_path)
#     im = tf.image.decode_png(im, channels=0)
#     im = tf.image.convert_image_dtype(im, dtype=tf.float32)
#     im2 = cv.imread(im_path, flags=0).astype("float32") /255
#     o1 = cv.dct(im.eval())
#     o2 = cv.dct(im2)
#
#     im_dct = tf.read_file(filename=im_dct_path)
#     im_dct = tf.image.decode_png(im_dct, channels=0)
#     im_dct = tf.image.convert_image_dtype(im_dct, dtype=tf.float32)
#     w1 = cv.dct(im_dct.eval())

train_path = "/home/wangruowei/PycharmProjects/watermark2020/data/gray/train"
test_path = "/home/wangruowei/PycharmProjects/watermark2020/data/gray/test"
watermarked_path = "/home/wangruowei/PycharmProjects/watermark2020/data/watermark_dct/"

with tf.Session() as sess:
    # i = im.eval()
    # i_w = im_dct.eval()
    # i_dct = cv.dct(i)
    # i_w_dct = cv.dct(i_w)
    # a = i_dct - i_w_dct
    # max = np.max(a)
    # min = np.min(a)

    max_train = -512
    min_train = 512
    max_count = 0
    min_count = 0
    i = 0
    for name in os.listdir(train_path):
        i = i + 1
        print(i)

        im_name = os.path.join(train_path, name)
        im = cv.imread(im_name, flags=0)
        im_f = im.astype('float32') / 255
        o = cv.dct(im_f)


        im_dct_name = os.path.join(watermarked_path, name.split(".")[0] + "_wDCT.png")
        im_dct = cv.imread(im_dct_name, flags=0)
        im_dct_f = im_dct.astype('float32') / 255
        w = cv.dct(im_dct_f)

        cha = o
        # cha = o - w
        # cha[0, 0] = 0

        if np.max(cha) > 1:
            max_count = max_count + 1
        if np.min(cha) < -1:
            min_count = min_count + 1

        if max_train < np.max(cha):
            max_train = np.max(cha)
        if min_train > np.min(cha):
            min_train = np.min(cha)

        cha = w
        if max_train < np.max(cha):
            max_train = np.max(cha)
        if min_train > np.min(cha):
            min_train = np.min(cha)

    max_test = -512
    min_test = 512
    for name in os.listdir(test_path):
        i = i + 1
        print(i)

        im_name2 = os.path.join(test_path, name)
        im2 = cv.imread(im_name2, flags=0)
        im_f2 = im2.astype('float32') / 255
        o2 = cv.dct(im_f2)

        im_dct_name2 = os.path.join(watermarked_path, name.split(".")[0] + "_wDCT.png")
        im_dct2 = cv.imread(im_dct_name2, flags=0)
        im_dct_f2 = im_dct2.astype('float32') / 255
        w2 = cv.dct(im_dct_f2)

        cha2 = o2
        # cha2 = o2 - w2
        # cha2[0, 0] = 0

        if np.max(cha) > 1:
            max_count = max_count + 1
        if np.min(cha) < -1:
            min_count = min_count + 1

        if max_test < np.max(cha2):
            max_test = np.max(cha2)
        if min_test > np.min(cha2):
            min_test = np.min(cha2)

        cha2 = w2
        if max_test < np.max(cha2):
            max_test = np.max(cha2)
        if min_test > np.min(cha2):
            min_test = np.min(cha2)

    max = max_train
    min = min_train
    if max < max_test:
        max = max_test
    if min > min_test:
        min = min_test

print("max_count: " + str(max_count))
print("min_count: " + str(min_count))
print("finished")
