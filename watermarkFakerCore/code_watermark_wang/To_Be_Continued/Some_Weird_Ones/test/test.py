import cv2 as cv
import tensorflow as tf
import numpy as np
import os


# image_path = "/home/wangruowei/PycharmProjects/watermark2020/code_watermark/dft_watermark/lena_grey_256.png"
# im_path = "/home/wangruowei/PycharmProjects/watermark2020/data/gray/test/007_0007.png"
# im_dct_path = "/home/wangruowei/PycharmProjects/watermark2020/data/watermark_dct/007_0007_wDCT.png"

# image = cv.imread(image_path, flags=0)
# # image = image.astype('float32')
# # x = cv.dct(image)
# image = image.astype('float32')
# image = tf.image.decode_png()
# image_01 = image / 255

# image_tf = tf.read_file(filename=image_path)
# image_tf = tf.image.decode_png(image_tf, channels=0)
# image_tf = tf.image.convert_image_dtype(image_tf, dtype=tf.float32)
#
# im = tf.read_file(filename=im_path)
# im = tf.image.decode_png(im, channels=0)
# im = tf.image.convert_image_dtype(im, dtype=tf.float32)
#
# im_dct = tf.read_file(filename=im_dct_path)
# im_dct = tf.image.decode_png(im_dct, channels=0)
# im_dct = tf.image.convert_image_dtype(im_dct, dtype=tf.float32)

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
    min_train = -512
    i = 0
    for name in os.listdir(train_path):
        i = i + 1
        print(i)

        im = tf.read_file(filename=os.path.join(train_path, name))
        im = tf.image.decode_png(im, channels=0)
        im = tf.image.convert_image_dtype(im, dtype=tf.float32)
        o = im.eval()

        im_dct = tf.read_file(filename=os.path.join(watermarked_path, name.split(".")[0] + "_wDCT.png"))
        im_dct = tf.image.decode_png(im_dct, channels=0)
        im_dct = tf.image.convert_image_dtype(im_dct, dtype=tf.float32)
        w = im_dct.eval()

        cha = o - w
        if max_train < np.max(cha):
            max_train = np.max(cha)
        if min_train > np.min(cha):
            min_train = np.min(cha)

    max_test = -512
    min_test = -512
    for name in os.listdir(test_path):
        i = i + 1
        print(i)

        im = tf.read_file(filename=os.path.join(test_path, name))
        im = tf.image.decode_png(im, channels=0)
        im = tf.image.convert_image_dtype(im, dtype=tf.float32)
        o = im.eval()

        im_dct = tf.read_file(filename=os.path.join(watermarked_path, name.split(".")[0] + "_wDCT.png"))
        im_dct = tf.image.decode_png(im_dct, channels=0)
        im_dct = tf.image.convert_image_dtype(im_dct, dtype=tf.float32)
        w = im_dct.eval()

        cha = o - w
        if max_test < np.max(cha):
            max_test = np.max(cha)
        if min_test > np.min(cha):
            min_test = np.min(cha)

    max = max_train
    min = max_test
    if max < max_test:
        max = max_test
    if min > min_test:
        min = min_test
    # shape1 = tf.shape(image_tf)
    # shape12 = shape1[2].eval()
    # x = image_tf.eval()
    # y_one = cv.dct(np.ones([256, 256]))
    # y = cv.dct(x)
    # y_n = (y + 512) / 1024
    # y_dct = cv.dct(x)
    # x_i = cv.idct(y)
    # y_reshape = np.reshape(y, [1, 256, 256, 1])
    # yy_reshape = np.append(y_reshape, y_reshape, axis=0)
    # print(y_reshape.shape[0])
    # z = tf.convert_to_tensor(y_reshape, dtype=tf.float32)
    # z_tf = z.eval()
    # # u = tf.image.convert_image_dtype(z, dtype=tf.float32)
    # # u_tf = u.eval()
    # # shape2 = tf.shape(u)
    # # # shape22 = shape2[2].eval()
    # # v = u.eval()
    # z_r = z_tf * 1024 - 512
    # # t_i = cv.idct(z_r)

print("finished")
