import cv2 as cv
import tensorflow as tf
from tensorflow import transpose
from tensorflow.spectral import dct, idct
import numpy as np
import os
from tqdm import tqdm


image_path = "/home/wangruowei/PycharmProjects/watermark2020/code_watermark/dft_watermark/lena_grey_256.png"

'''opencv版本'''
# image = cv.imread(image_path, flags=0)
# image = image.astype('float32') / 255
# x = cv.dct(image)
# x_2 = x / 10
# x_2[0, 0] = 1
# tmp = x_2 * 10
# y = cv.idct(tmp)
# y_save = y * 255
# cv.imwrite("y.png", y_save)
'''-----分界线-----'''

'''tensorflow版本'''
image_tf = tf.read_file(filename=image_path)
image_tf = tf.image.decode_png(image_tf, channels=0)
image_tf_float = tf.image.convert_image_dtype(image_tf, dtype=tf.float32)
image_tf_float_255 = tf.to_float(image_tf)

with tf.Session() as sess:
    '''tf图像读取实验'''
    int8 = image_tf.eval()
    float32 = image_tf_float.eval()
    float32_255 = image_tf_float_255.eval()
    '''-----分割线-----'''

    sess.run(tf.global_variables_initializer())

    '''block-dct实验'''
    def block_dct(image):
        # 分块，然后进行DCT变换
        dct_image = 0
        for i in tqdm(range(32)):
            row = 0
            for j in range(32):
                block = image[i*8:(i+1)*8, j*8:(j+1)*8]
                block = dct(transpose(dct(transpose(block), norm='ortho')), norm='ortho')
                if j == 0:
                    row = block
                else:
                    row = tf.concat([row, block], 1)
            if i == 0:
                dct_image = row
            else:
                dct_image = tf.concat([dct_image, row], 0)
        return dct_image

    def block_idct(image):
        # 分块，然后进行DCT变换
        dct_image = 0
        for i in tqdm(range(32)):
            row = 0
            for j in range(32):
                block = image[i*8:(i+1)*8, j*8:(j+1)*8]
                block = idct(transpose(idct(transpose(block), norm='ortho')), norm='ortho')
                if j == 0:
                    row = block
                else:
                    row = tf.concat([row, block], 1)
            if i == 0:
                dct_image = row
            else:
                dct_image = tf.concat([dct_image, row], 0)
        return dct_image

    a = image_tf_float_255
    a_dct = block_dct(a)
    a_dct_idct = block_idct(a_dct)

    # sess.run(op)
    av = a.eval()
    a_dctv = a_dct.eval()
    a_dct_idctv = a_dct_idct.eval()

    cv = av - a_dct_idctv
    pass

    # a = float32_255
    # a = tf.reshape(a, [256, 256, 1])
    # a_dct = tf.transpose(tf.spectral.dct(tf.transpose(tf.spectral.dct(a, norm='ortho')), norm='ortho'))
    # a_dctv = a_dct.eval()
    # a_dct2 = tf.spectral.dct(tf.transpose(tf.spectral.dct(tf.transpose(a), norm='ortho')), norm='ortho')
    # a_dct2v = a_dct2.eval()
    #
    # a_i1 = tf.transpose(tf.spectral.idct(tf.transpose(tf.spectral.idct(a_dct, norm='ortho')), norm='ortho'))
    # a_i1 = a_i1.eval()
    # a_i2 = tf.spectral.idct(tf.transpose(tf.spectral.idct(tf.transpose(a_dct2), norm='ortho')), norm='ortho')
    # a_i2 = a_i2.eval()
    #
    # b_dct = cv.dct(tf.reshape(a, [4, 4]).eval())
    # b_i = cv.idct(b_dct)
    '''-----分界线-----'''

    '''多维切分实验'''
    # a = tf.constant(np.arange(64), dtype=tf.float32)
    # a = tf.reshape(a, [4, 4, 4, 1])
    # b = tf.reshape(a[0], [1, 4, 4, 1])
    # c = tf.reshape(a[1], [1, 4, 4, 1])
    # a1 = a[0].eval()
    # a2 = a[1].eval()
    # a3 = a[2].eval()
    # a4 = a[3].eval()
    # a_v = a.eval()
    # b = b.eval()
    # x = tf.concat([b, c], axis=0)
    # x_v = x.eval()
    '''-----分割线-----'''

    '''二维版本的DCT逆变换测试成功，无论是先行后列，还是先列后行，两者都一致'''
    # a = tf.constant(np.arange(16), dtype=tf.float32)
    # a = tf.reshape(a, [4, 4, 1])
    # a_dct = tf.transpose(tf.spectral.dct(tf.transpose(tf.spectral.dct(a, norm='ortho')), norm='ortho'))
    # a_dctv = a_dct.eval()
    # a_dct2 = tf.spectral.dct(tf.transpose(tf.spectral.dct(tf.transpose(a), norm='ortho')), norm='ortho')
    # a_dct2v = a_dct2.eval()
    #
    # a_i1 = tf.transpose(tf.spectral.idct(tf.transpose(tf.spectral.idct(a_dct, norm='ortho')), norm='ortho'))
    # a_i1 = a_i1.eval()
    # a_i2 = tf.spectral.idct(tf.transpose(tf.spectral.idct(tf.transpose(a_dct2), norm='ortho')), norm='ortho')
    # a_i2 = a_i2.eval()
    #
    # b_dct = cv.dct(tf.reshape(a, [4, 4]).eval())
    # b_i = cv.idct(b_dct)
    '''-----分界线-----'''

    '''一维版本的DCT逆变换测试测试成功，两者基本一致'''
    # a = np.arange(10, dtype=np.float32) / 9
    # b = cv.dct(a)
    # b_i = cv.idct(b)
    # c = tf.convert_to_tensor(a)
    # print(c.eval())
    # d = tf.spectral.dct(c, norm='ortho')
    # d = d.eval()
    # d_i = tf.spectral.idct(d, norm='ortho')
    # d_i = d_i.eval()
    '''-----分界线-----'''

    '''二维版本的DCT正变换测试成功，无论是先行后列，还是先列后行，两者都一致'''
    # a = tf.constant(np.arange(16), dtype=tf.float32)
    # a = tf.reshape(a, [4, 4, 1])
    # a_dct = tf.transpose(tf.spectral.dct(tf.transpose(tf.spectral.dct(a, norm='ortho')), norm='ortho'))
    # a_dct = a_dct.eval()
    # a_dct2 = tf.spectral.dct(tf.transpose(tf.spectral.dct(tf.transpose(a), norm='ortho')), norm='ortho')
    # a_dct2 = a_dct2.eval()
    # b_dct = cv.dct(a.eval())
    # print(a.eval())
    # print(a_dct)
    # print(b_dct)
    '''-----分界线-----'''

    '''一维版本的DCT正变换测试成功，两者基本一致'''
    # a = np.arange(10, dtype=np.float32) / 9
    # b = cv.dct(a)
    # b_i = cv.idct(b)
    # c = tf.convert_to_tensor(a)
    # print(c.eval())
    # d = tf.spectral.dct(c, norm='ortho')
    # c = d.eval()
    '''-----分界线-----'''

    pass
'''-----分界线-----'''

print("finished")
