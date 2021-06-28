import tensorflow as tf
import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math
import time

# https://blog.csdn.net/MOU_IT/article/details/80802407，该网页中的部分实现参考了该pix2pix的代码，里面有些注释，可以参考参考。

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", default="D:\CS-Related\Watermark Faker\Test_Images", help="path to folder containing images")
parser.add_argument("--mode", required=False, default="train", choices=["train", "test", "export"])
parser.add_argument("--output_dir", required=False, default="D:\CS-Related\Watermark Faker\Watermark Faker Output", help="where to put output files")
parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing")

parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, default=200, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=500, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=10, help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=500, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=100, help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=100, help="save model every save_freq steps, 0 to disable")

parser.add_argument("--separable_conv", action="store_true", help="use separable convolutions in the generator")
parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")
parser.add_argument("--lab_colorization", action="store_true", help="split input image into brightness (A) and color (B)")
parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
parser.add_argument("--which_direction", type=str, default="AtoB", choices=["AtoB", "BtoA"])
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
parser.add_argument("--scale_size", type=int, default=256, help="scale images to this size before cropping to 256x256")
parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
parser.set_defaults(flip=False)
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")

# export options
parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])
a = parser.parse_args()

EPS = 1e-12
CROP_SIZE = 256  # 原始数值：256

def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1

def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2

def preprocess_lab(lab):
    with tf.name_scope("preprocess_lab"):
        L_chan, a_chan, b_chan = tf.unstack(lab, axis=2)
        # L_chan: black and white with input range [0, 100]
        # a_chan/b_chan: color channels with input range ~[-110, 110], not exact
        # [0, 100] => [-1, 1],  ~[-110, 110] => [-1, 1]
        return [L_chan / 50 - 1, a_chan / 110, b_chan / 110]

def deprocess_lab(L_chan, a_chan, b_chan):
    with tf.name_scope("deprocess_lab"):
        # this is axis=3 instead of axis=2 because we process individual images but deprocess batches
        return tf.stack([(L_chan + 1) / 2 * 100, a_chan * 110, b_chan * 110], axis=3)

def augment(image, brightness):
    # (a, b) color channels, combine with L channel and convert to rgb
    a_chan, b_chan = tf.unstack(image, axis=3)
    L_chan = tf.squeeze(brightness, axis=3)
    lab = deprocess_lab(L_chan, a_chan, b_chan)
    rgb = lab_to_rgb(lab)
    return rgb


# 构建 D 的卷积层扩充一圈0，而不是像 G 的卷积层那样，使用padding="same"参数？
# 疑问：为什么这里要手动
def discrim_conv(batch_input, out_channels, stride):
    # 下一行的 tf.pad()表示只给图片的长、宽周围垫一圈0，而不管batch、channel，参见：https://blog.csdn.net/qq_40994943/article/details/85331327
    padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    return tf.layers.conv2d(padded_input, out_channels, kernel_size=4, strides=(stride, stride), padding="valid", kernel_initializer=tf.random_normal_initializer(0, 0.02))


# 构建 G 的卷积层
# 根据输入的tensor(对应这里的batch_input)和需要输出的层数(对应这里的out_channels)，创建卷积层，并返回卷积后的结果
def gen_conv(batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)  # 返回一个生成具有正态分布的张量的初始化器。参见：https://blog.csdn.net/weixin_34252686/article/details/89804826，https://www.w3cschool.cn/tensorflow_python/tensorflow_python-b8jq2gqh.html
    if a.separable_conv:  # 疑问：作用不明，非最优先项，暂时忽略
        return tf.layers.separable_conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", depthwise_initializer=initializer, pointwise_initializer=initializer)
    else:
        # tf.layers.conv2d()，参见：https://blog.csdn.net/gqixf/article/details/80519912
        return tf.layers.conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer)


def gen_deconv(batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    if a.separable_conv:
        _b, h, w, _c = batch_input.shape
        resized_input = tf.image.resize_images(batch_input, [h * 2, w * 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return tf.layers.separable_conv2d(resized_input, out_channels, kernel_size=4, strides=(1, 1), padding="same", depthwise_initializer=initializer, pointwise_initializer=initializer)
    else:
        return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer)


# a为斜率参数，该函数为标准的leakyRelu，参见：https://blog.csdn.net/sinat_33027857/article/details/80192789
def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(inputs):
    return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))


def check_image(image):
    # 检查图像是否是3通道彩色图
    assertion = tf.assert_equal(tf.shape(image)[-1], 3, message="image must have 3 color channels")
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    # get_shape().ndims返回了维度的个数，如[h, w, c]的ndims为3，参见：https://www.jianshu.com/p/653096c9defe
    # 用于检验维度个数是否正确
    if image.get_shape().ndims not in (3, 4):
        raise ValueError("image must be either 3 or 4 dimensions")

    # make the last dimension 3 so that you can unstack the colors
    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image


# based on https://github.com/torch/image/blob/9f65c30167b2048ecbe8b7befdc6b2d6d12baee9/generic/image.c
def rgb_to_lab(srgb):
    # 待解析···

    with tf.name_scope("rgb_to_lab"):
        srgb = check_image(srgb)  # 检查image的通道数和维数，并修正shape，将通道数设置为3，方便后买你的计算
        srgb_pixels = tf.reshape(srgb, [-1, 3])  # -1的解释，在该函数的官方解释中有，用于flatten像素，这里具体的意思为，将三个通道上的值从矩阵变为单独一行，最终得到3行。

        with tf.name_scope("srgb_to_xyz"):
            linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
            exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
            rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
            rgb_to_xyz = tf.constant([
                #    X        Y          Z
                [0.412453, 0.212671, 0.019334],  # R
                [0.357580, 0.715160, 0.119193],  # G
                [0.180423, 0.072169, 0.950227],  # B
            ])
            xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("xyz_to_cielab"):
            # convert to fx = f(X/Xn), fy = f(Y/Yn), fz = f(Z/Zn)

            # normalize for D65 white point
            xyz_normalized_pixels = tf.multiply(xyz_pixels, [1/0.950456, 1.0, 1/1.088754])

            epsilon = 6/29
            linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon**3), dtype=tf.float32)
            exponential_mask = tf.cast(xyz_normalized_pixels > (epsilon**3), dtype=tf.float32)
            fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4/29) * linear_mask + (xyz_normalized_pixels ** (1/3)) * exponential_mask

            # convert to lab
            fxfyfz_to_lab = tf.constant([
                #  l       a       b
                [  0.0,  500.0,    0.0], # fx
                [116.0, -500.0,  200.0], # fy
                [  0.0,    0.0, -200.0], # fz
            ])
            lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])

        return tf.reshape(lab_pixels, tf.shape(srgb))


def lab_to_rgb(lab):
    with tf.name_scope("lab_to_rgb"):
        lab = check_image(lab)
        lab_pixels = tf.reshape(lab, [-1, 3])

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("cielab_to_xyz"):
            # convert to fxfyfz
            lab_to_fxfyfz = tf.constant([
                #   fx      fy        fz
                [1/116.0, 1/116.0,  1/116.0], # l
                [1/500.0,     0.0,      0.0], # a
                [    0.0,     0.0, -1/200.0], # b
            ])
            fxfyfz_pixels = tf.matmul(lab_pixels + tf.constant([16.0, 0.0, 0.0]), lab_to_fxfyfz)

            # convert to xyz
            epsilon = 6/29
            linear_mask = tf.cast(fxfyfz_pixels <= epsilon, dtype=tf.float32)
            exponential_mask = tf.cast(fxfyfz_pixels > epsilon, dtype=tf.float32)
            xyz_pixels = (3 * epsilon**2 * (fxfyfz_pixels - 4/29)) * linear_mask + (fxfyfz_pixels ** 3) * exponential_mask

            # denormalize for D65 white point
            xyz_pixels = tf.multiply(xyz_pixels, [0.950456, 1.0, 1.088754])

        with tf.name_scope("xyz_to_srgb"):
            xyz_to_rgb = tf.constant([
                #     r           g          b
                [ 3.2404542, -0.9692660,  0.0556434], # x
                [-1.5371385,  1.8760108, -0.2040259], # y
                [-0.4985314,  0.0415560,  1.0572252], # z
            ])
            rgb_pixels = tf.matmul(xyz_pixels, xyz_to_rgb)
            # avoid a slightly negative number messing up the conversion
            rgb_pixels = tf.clip_by_value(rgb_pixels, 0.0, 1.0)
            linear_mask = tf.cast(rgb_pixels <= 0.0031308, dtype=tf.float32)
            exponential_mask = tf.cast(rgb_pixels > 0.0031308, dtype=tf.float32)
            srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + ((rgb_pixels ** (1/2.4) * 1.055) - 0.055) * exponential_mask

        return tf.reshape(srgb_pixels, tf.shape(lab))


def load_examples():
    # 检查输入文件夹是否为空，或不存在
    if a.input_dir is None or not os.path.exists(a.input_dir):
        raise Exception("input_dir does not exist")

    # 读取所有输入图像的路径，并决定解码方式
    # input_paths包含了所有的input的具体路径
    # decode为针对不同存储格式的图片的解码方式
    input_paths = glob.glob(os.path.join(a.input_dir, "*.jpg"))  # glop.glop()用于获取符合输入路径格式的所有文件的具体路径，（包括文件夹和文件），参见：https://www.cnblogs.com/luminousjj/p/9359543.html
    decode = tf.image.decode_jpeg
    if len(input_paths) == 0:
        input_paths = glob.glob(os.path.join(a.input_dir, "*.png"))
        decode = tf.image.decode_png
    '''以下修改，by王爇沩'''
    if len(input_paths) == 0:
        input_paths = glob.glob(os.path.join(a.input_dir, "*.bmp"))
        decode = tf.image.decode_bmp
    '''分割线'''
    if len(input_paths) == 0:
        raise Exception("input_dir contains no image files")

    # 取得文件的名字（不含拓展名，如.txt）
    def get_name(path):
        # basename()返回路径末尾的文件名，如code.py。参见：https://www.cnblogs.com/baxianhua/p/10214263.html
        # splitext()将文件名和拓展名分开。参见：https://blog.csdn.net/T1243_3/article/details/80170006
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    # 根据文件的名字是字符还是纯数字进行重排序
    # if all the image names are numbers, sort by the value rather than ascetically
    # having sorted inputs means that the outputs are sorted in test mode
    if all(get_name(path).isdigit() for path in input_paths):
        input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
    else:
        input_paths = sorted(input_paths)

    with tf.name_scope("load_images"):  # name_scope() 只会对节点的 name 产生影响；不会影响作用域，参见：https://www.jianshu.com/p/635d95b34e14
        # 读取图片，解码图片，并将其归一化到 [-1, 1]
        path_queue = tf.train.string_input_producer(input_paths, shuffle=a.mode == "train")  # 将一堆文件名整合成一个python queue（队列）
        reader = tf.WholeFileReader()  # 创建reader用于读取上一行创建的队列，参见：https://blog.csdn.net/xuan_zizizi/article/details/78418351, https://blog.csdn.net/houyanhua1/article/details/88194016
        paths, contents = reader.read(path_queue)  # 用reader读取queue，paths为文件路径（包含最后的文件名），contents为图片的实际内容，参见：https://blog.csdn.net/xuan_zizizi/article/details/78418351
        raw_input = decode(contents)  # 对批量图片进行解码，得到图片原本的tensor。这样的decode得到的格式为uint8格式。
        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)  # 将图片的像素值转换为区间在0-1之间的float32格式（归一化）

        # 当图片是3通道时，才执行with下一行的语句。疑问：这里的tf.identity的作用尚不明晰，大致意思为将一个tensor转换为了op
        # 参见：https://www.jianshu.com/p/9de22c907795 https://www.cnblogs.com/hellcat/p/8568035.html
        # 参见：https://www.jianshu.com/p/1938a958d986 https://blog.csdn.net/fyq201749/article/details/82118013
        assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")  # 参见：https://blog.csdn.net/fyq201749/article/details/82118013
        with tf.control_dependencies([assertion]):
            raw_input = tf.identity(raw_input)

        # raw_input在被计算之前，是没有shape的，这里将其赋予了3，表示该图片是三通道的（这个由正上方的代码保证）。
        raw_input.set_shape([None, None, 3])

        if a.lab_colorization:
            # load color and brightness from image, no B image exists here
            lab = rgb_to_lab(raw_input)
            L_chan, a_chan, b_chan = preprocess_lab(lab)
            a_images = tf.expand_dims(L_chan, axis=2)
            b_images = tf.stack([a_chan, b_chan], axis=2)
        else:
            # break apart image pair and move to range [-1, 1]
            width = tf.shape(raw_input)[1]  # [height, width, channels]
            a_images = preprocess(raw_input[:, :width//2, :])  # ‘//’整数除法，这里是将一对图片的左边，切下来，并归一化到 [-1, 1]。下面是对右半的图像做处理
            b_images = preprocess(raw_input[:, width//2:, :])

    # 根据方向，设置输入、输出
    if a.which_direction == "AtoB":
        inputs, targets = [a_images, b_images]
    elif a.which_direction == "BtoA":
        inputs, targets = [b_images, a_images]
    else:
        raise Exception("invalid direction")

    # synchronize seed for image operations so that we do the same operations to both
    # input and output images
    seed = random.randint(0, 2**31 - 1)
    Examples = collections.namedtuple("Examples", "paths, inputs, targets, count, steps_per_epoch")
    # 根据参数设置，决定是否对图片进行反转、放大剪裁
    def transform(image):
        r = image
        if a.flip:  # 需要将图片反转的话，就左右翻转
            r = tf.image.random_flip_left_right(r, seed=seed)

        # area produces a nice downscaling, but does nearest neighbor for upscaling
        # assume we're going to be doing downscaling here
        r = tf.image.resize_images(r, [a.scale_size, a.scale_size], method=tf.image.ResizeMethod.AREA)

        # tf.cast()用于转换数据类型
        # tf.floor()向下取整
        offset = tf.cast(tf.floor(tf.random_uniform([2], 0, a.scale_size - CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
        if a.scale_size > CROP_SIZE:
            r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], CROP_SIZE, CROP_SIZE)  # 设置起点、剪裁大小，然后剪裁图片
        elif a.scale_size < CROP_SIZE:
            raise Exception("scale size cannot be less than crop size")
        return r

    with tf.name_scope("input_images"):
        input_images = transform(inputs)

    with tf.name_scope("target_images"):
        target_images = transform(targets)

    # 使输入的tensor能够以batch的形式运行并输出，参见：https://blog.csdn.net/sinat_29957455/article/details/83152823
    # 疑问：tf.train.batch()函数的作用大概明白了，更深一层的理解，还需实战
    paths_batch, inputs_batch, targets_batch = tf.train.batch([paths, input_images, target_images], batch_size=a.batch_size)
    steps_per_epoch = int(math.ceil(len(input_paths) / a.batch_size))  # 计算每个epoch至少需要的step数

    return Examples(
        paths=paths_batch,
        inputs=inputs_batch,
        targets=targets_batch,
        count=len(input_paths),
        steps_per_epoch=steps_per_epoch,
    )


def create_generator(generator_inputs, generator_outputs_channels):
    layers = []

    # 创建 G 的编码部分的第一层
    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("encoder_1"):
        output = gen_conv(generator_inputs, a.ngf)  # 将input1输入第一层，得到对应的输出output1
        layers.append(output)  # 将output1保存

    # 设定好不同的编码层的输出
    layer_specs = [
        a.ngf * 2,  # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        a.ngf * 4,  # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        a.ngf * 8,  # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        a.ngf * 8,  # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        a.ngf * 8,  # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        a.ngf * 8,  # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        a.ngf * 8,  # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]

    # 创建 G 的编码部分的剩下7层
    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)  # 上一层的输出，经过激活函数处理
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = gen_conv(rectified, out_channels)  # 上一层的输出，进入下一层
            output = batchnorm(convolved)  # 当前层的输出，进行batchnorm归一化
            layers.append(output)  # 将当前层的输出记录下来

    # 设定好不同的解码层的输出
    layer_specs = [
        (a.ngf * 8, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (a.ngf * 8, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (a.ngf * 8, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (a.ngf * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (a.ngf * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (a.ngf * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (a.ngf, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    ]

    # 获取编码层的个数
    num_encoder_layers = len(layers)

    # 构造 7 个编码层，的其对应的输出
    # enumerate()函数，参见：https://www.runoob.com/python/python-func-enumerate.html
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1  # 获取当前解码层对应的编码层的序号，用以获取编码层的输出
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            # 根据解码层的序号，结合skip_connection，设置输入
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)  # tf.concat()可以简单理解为在指定维度上发生叠加，参见：https://www.jianshu.com/p/c465e7fab882

            rectified = tf.nn.relu(input)  # 典型的Relu函数，参见：https://www.cnblogs.com/xh_chiang/p/9132524.html
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = gen_deconv(rectified, out_channels)
            output = batchnorm(output)

            # dropout函数，参见：https://blog.csdn.net/yangfengling1023/article/details/82911306
            # 疑问：为什么这里用上了dropout？
            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # 构造最后一层编码层，得到其输出
    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        output = gen_deconv(rectified, generator_outputs_channels)
        output = tf.tanh(output)
        layers.append(output)

    return layers[-1]


def create_model(inputs, targets):
    def create_discriminator(discrim_inputs, discrim_targets):
        n_layers = 3
        layers = []  # 储存每一层的输出

        # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
        # 将input与另外一个输入（output或target）拼接在一起
        input = tf.concat([discrim_inputs, discrim_targets], axis=3)

        # 构建 D 的第一层
        # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
        with tf.variable_scope("layer_1"):
            convolved = discrim_conv(input, a.ndf, stride=2)
            rectified = lrelu(convolved, 0.2)  # 疑问：为什么这里的leakyRelu的斜率参数被设置为0.2？
            layers.append(rectified)

        # 构建 D 的2到4层
        # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
        # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
        # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channels = a.ndf * min(2**(i+1), 8)  # 设定输出的层数
                stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1  # 设置每层的stride。第4层的stride会被设置为1，其余设置为2，这里的写法很有趣
                convolved = discrim_conv(layers[-1], out_channels, stride=stride)  # 进行卷积
                normalized = batchnorm(convolved)
                rectified = lrelu(normalized, 0.2)
                layers.append(rectified)

        # 构建 D 的最后一层，即第5层，注意：激活函数用的sigmoid
        # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            convolved = discrim_conv(rectified, out_channels=1, stride=1)
            output = tf.sigmoid(convolved)
            layers.append(output)

        # 将最后一层的结果，即 D 的整体结果返回
        return layers[-1]

    # 根据targets图片，取得 G 最后的输出通道个数。
    # 然后，创建根据输入和输出，创建生成器，并取得生成器G 的输出
    # 疑问：tf.variable_scope()的作用，还有待进一步调查，参见：https://blog.csdn.net/qq_22522663/article/details/78729029，https://blog.csdn.net/uestc_c2_403/article/details/72328815
    with tf.variable_scope("generator"):
        out_channels = int(targets.get_shape()[-1])  # out_channels根据图片是彩色还是灰度图，决定最后输出的图片的通道数
        outputs = create_generator(inputs, out_channels)  # outputs即为 G 的输出

    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables，参见：https://affinelayer.com/pix2pix/
    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_real = create_discriminator(inputs, targets)

    with tf.name_scope("fake_discriminator"):
        # 在这一行就体现了tf.variable_scope()的作用了（重用），两个discriminator实际共用一套参数
        with tf.variable_scope("discriminator", reuse=True):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_fake = create_discriminator(inputs, outputs)

    with tf.name_scope("discriminator_loss"):
        # minimizing -tf.log will try to get inputs to 1 (因为对于D来说，是要MAX，所以这里 miniminzing 负的tf.log)
        # predict_real => 1
        # predict_fake => 0
        # tf.reduce_mean()用于计算tensor的均值，参见：https://blog.csdn.net/dcrmg/article/details/79797826
        # tf.log()其实就是求tensor中，每个元素a的 ln_a 值，然后将其以原来的shape返回
        discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))  # eps可能是为了防止值为0而设置的。

    with tf.name_scope("generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))  # 这种版本的GANloss是正确的，其目的是为了解决训练初期的饱和问题，详见 https://www.chainnews.com/articles/042578835630.htm 中的4号公式
        gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
        gen_loss = gen_loss_GAN * a.gan_weight + gen_loss_L1 * a.l1_weight

    # 根据 D 的loss结果，调整参数
    with tf.name_scope("discriminator_train"):
        # tf.trainable_variables()取得所有可训练的参数，参见：https://blog.csdn.net/cerisier/article/details/86523446
        # 该语句用于取得所有名字以"discriminator"开头的变量，即 D 的所有参数
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]

        # Adam优化算法，参见：https://blog.csdn.net/TeFuirnever/article/details/88933368
        # 疑问：具体的优化细节还有待进一步学习
        discrim_optim = tf.train.AdamOptimizer(a.lr, a.beta1)

        # 计算梯度，并应用梯度，参见：https://blog.csdn.net/shenxiaoming77/article/details/77169756，https://blog.csdn.net/sinat_37386947/article/details/88849519
        # 疑问：感觉要简单的撰写个y=w*x+b的例子，才能很好的理解的样子，有空的话，搞一搞
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)  # 将计算出的梯度应用到变量上，是函数minimize()的第二部分，返回一个应用指定的梯度的操作Operation，对global_step做自增操作，参见：https://blog.csdn.net/shenxiaoming77/article/details/77169756

    # 根据 G 的loss结果，调整参数
    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):  # 这里应该是表示，D 训练后，才能对 G 进行训练，疑问：具体意思还需要进一步理解
            # 与 D 的调整参数部分，同理
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    # 关于tf.train.ExponentialMovingAverage()，参见：https://www.jianshu.com/p/2f53606d4b6d，https://blog.csdn.net/tefuirnever/article/details/88902132
    # 关于影子变量的补充说明：原始变量为v，影子变量为v_s。每次使用ema.apply()时，会根据当前的v和decay，修改v_s的值。ema.average()函数用于获取当前的v_s值。
    # ————所以，v_s相当于，v的多次加权平均和。每次v_s的更新，可以反映当前v的变化情况。v_s一旦创建，会一直保存。
    # 关于decay的补充说明：decay可能存在变化，这个与step相关。
    ema = tf.train.ExponentialMovingAverage(decay=0.99)  # 创建一个EMA
    # ema.apply()大体意思为：对括号中的变量，依据decay更新其对应的影子变量（如果没有影子变量就创建）。apply()是一个随时可以执行的操作，即op
    update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])

    # 创建或返回一个全局步数的 tensor。参见：https://blog.csdn.net/a1054513777/article/details/79611801
    global_step = tf.train.get_or_create_global_step()

    # global_step加1，这是一个可以重复执行的操作op，参见：https://blog.csdn.net/a19990412/article/details/82917734
    incr_global_step = tf.assign(global_step, global_step+1)
    Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, train")
    return Model(
        predict_real=predict_real,
        predict_fake=predict_fake,
        discrim_loss=ema.average(discrim_loss),  # 取得discrim_loss的影子变量的值
        discrim_grads_and_vars=discrim_grads_and_vars,
        gen_loss_GAN=ema.average(gen_loss_GAN),  # 取得gen_loss_GAN的影子变量的值
        gen_loss_L1=ema.average(gen_loss_L1),  # 取得gen_loss_L1的影子变量的值
        gen_grads_and_vars=gen_grads_and_vars,
        outputs=outputs,  # G 生成的图片
        # 直接用ctrl查看tf.group()的官方解释，tf.group()返回的是An Operation that executes all its inputs.
        # ————也就是说得到一个可以同时执行所有操作的另一个op
        # 由于有限定关系，在执行gen_train前，必须先执行discrim_train，所以这里是G和D交替发生了训练，每个训练一次
        train=tf.group(update_losses, incr_global_step, gen_train),
    )


# 保存训练途中
def save_images(fetches, step=None):
    # 设定图片存储路径
    image_dir = os.path.join(a.output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    # 因为train时，每次输入该函数的是一个batch的图片，所以碰到需要display时，每次只会保存并展示batch张图片
    # 而test时，……（由于test部分还每解析，所以待写。但大体意思应该是没错的，test多次调用了这个函数，使得所有图片都被处理了）
    # 下面部分的模块的意思为，fetches的某张图片对应的input,output,target写入输出文件
    # 并在最后返回一个列表，列表的每个单元为字典，其内容如下{name, step, input_name, output_name, target_name}
    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))  # 疑问：这里的decode()还不彻底懂，暂时忽略
        fileset = {"name": name, "step": step}
        for kind in ["inputs", "outputs", "targets"]:
            filename = name + "-" + kind + ".png"  # 原始为 png。处理bmp图片时，可以改为bmp
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]  # 这里取出对应的解码好的图片
            with open(out_path, "wb") as f:
                f.write(contents)
        filesets.append(fileset)
    return filesets


def append_index(filesets, step=False):
    # 将display得到的图片名字，对应的step，写入index。打开该index后，会自动索引得到图片的内容
    # 返回index的路径
    index_path = os.path.join(a.output_dir, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        index.write("<th>name</th><th>input</th><th>output</th><th>target</th></tr>")

    for fileset in filesets:
        index.write("<tr>")

        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s</td>" % fileset["name"])

        for kind in ["inputs", "outputs", "targets"]:
            index.write("<td><img src='images/%s'></td>" % fileset[kind])

        index.write("</tr>")
    return index_path


def main():
    if a.seed is None:
        a.seed = random.randint(0, 2**31 - 1)  # 如果命令行输入是没有确定seed的值，那么产生随机数的种子

    tf.set_random_seed(a.seed)  # 设置tensorflow的随机数种子，参考：https://blog.csdn.net/qq_31878983/article/details/79495810
    np.random.seed(a.seed)  # 设置numpy的随机数种子，参考：https://www.cnblogs.com/subic/p/8454025.html
    random.seed(a.seed)  # 设置random模块的随机数种子

    if not os.path.exists(a.output_dir):  # 输出目录不存在时，就创建输出目录
        os.makedirs(a.output_dir)


    # 该判断语句用于：在 test 或 export 模式时，读取设置的参数
    if a.mode == "test" or a.mode == "export":
        if a.checkpoint is None:  # 检查checkpoint是否存在
            raise Exception("checkpoint required for test mode")

        # load some options from the checkpoint
        options = {"which_direction", "ngf", "ndf", "lab_colorization"}
        with open(os.path.join(a.checkpoint, "options.json")) as f:  # 打开文件。os.path.join会自动在末尾加上‘/’，参见：https://www.cnblogs.com/an-ning0920/p/10037790.html
            for key, val in json.loads(f.read()).items():  # json.loads()将json类型的字符串转换为dic（字典），然后dic调用item()函数，以列表返回可遍历的(键, 值) 元组数组参见：https://www.cnblogs.com/hjianhui/p/10387057.html
                if key in options:
                    print("loaded", key, "=", val)
                    setattr(a, key, val)  # setattr()函数用于设置属性值，参见：https://www.runoob.com/python/python-func-setattr.html
        # disable these features in test mode
        a.scale_size = CROP_SIZE
        a.flip = False

    # 打印 a 中的各项声明、参数
    for k, v in a._get_kwargs():
        print(k, "=", v)

    # 将当前的 a 的设置，写入json文件中
    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))
        # vars()函数返回对象object的属性和属性值的字典对象，参见：https://www.runoob.com/python/python-func-vars.html。
        # json.dumps()将字典转换为json格式字符串，参见：https://www.cnblogs.com/hjianhui/p/10387057.html

    # 待解析··················
    if a.mode == "export":
        # export the generator to a meta graph that can be imported later for standalone generation
        if a.lab_colorization:
            raise Exception("export not supported for lab_colorization")

        input = tf.placeholder(tf.string, shape=[1])
        input_data = tf.decode_base64(input[0])
        input_image = tf.image.decode_png(input_data)

        # remove alpha channel if present
        input_image = tf.cond(tf.equal(tf.shape(input_image)[2], 4), lambda: input_image[:,:,:3], lambda: input_image)
        # convert grayscale to RGB
        input_image = tf.cond(tf.equal(tf.shape(input_image)[2], 1), lambda: tf.image.grayscale_to_rgb(input_image), lambda: input_image)

        input_image = tf.image.convert_image_dtype(input_image, dtype=tf.float32)
        input_image.set_shape([CROP_SIZE, CROP_SIZE, 3])
        batch_input = tf.expand_dims(input_image, axis=0)

        with tf.variable_scope("generator"):
            batch_output = deprocess(create_generator(preprocess(batch_input), 3))

        output_image = tf.image.convert_image_dtype(batch_output, dtype=tf.uint8)[0]
        if a.output_filetype == "png":
            output_data = tf.image.encode_png(output_image)
        elif a.output_filetype == "jpeg":
            output_data = tf.image.encode_jpeg(output_image, quality=80)
        else:
            raise Exception("invalid filetype")
        output = tf.convert_to_tensor([tf.encode_base64(output_data)])

        key = tf.placeholder(tf.string, shape=[1])
        inputs = {
            "key": key.name,
            "input": input.name
        }
        tf.add_to_collection("inputs", json.dumps(inputs))
        outputs = {
            "key":  tf.identity(key).name,
            "output": output.name,
        }
        tf.add_to_collection("outputs", json.dumps(outputs))

        init_op = tf.global_variables_initializer()
        restore_saver = tf.train.Saver()
        export_saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init_op)
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            restore_saver.restore(sess, checkpoint)
            print("exporting model")
            export_saver.export_meta_graph(filename=os.path.join(a.output_dir, "export.meta"))
            export_saver.save(sess, os.path.join(a.output_dir, "export"), write_meta_graph=False)

        return

    # 读取图片，对其解码，进行预处理（翻转，放大后剪裁一下；前面的操作是可选的），根据方向，决定input与output的分别是哪些图片，得到其路径
    # 还可以得到input，target图片集
    examples = load_examples()
    print("examples count = %d" % examples.count)

    # 创建模型，得到模型的输出output集合，训练操作op，等一系列参数
    # inputs and targets are [batch_size, height, width, channels]
    model = create_model(examples.inputs, examples.targets)

    # undo colorization splitting on images that we use for display/output
    if a.lab_colorization:  # 待解析：该分支目前未使用过，所以暂时不管它
        if a.which_direction == "AtoB":
            # inputs is brightness, this will be handled fine as a grayscale image
            # need to augment targets and outputs with brightness
            targets = augment(examples.targets, examples.inputs)
            outputs = augment(model.outputs, examples.inputs)
            # inputs can be deprocessed normally and handled as if they are single channel
            # grayscale images
            inputs = deprocess(examples.inputs)
        elif a.which_direction == "BtoA":
            # inputs will be color channels only, get brightness from targets
            inputs = augment(examples.inputs, examples.targets)
            targets = deprocess(examples.targets)
            outputs = deprocess(model.outputs)
        else:
            raise Exception("invalid direction")
    else:
        # 将所有图片的值归一化到[0, 1]
        inputs = deprocess(examples.inputs)
        targets = deprocess(examples.targets)
        outputs = deprocess(model.outputs)

    # 将image的像素值格式转换为uint8。
    # 将图片变回原来的比例（可选）
    def convert(image):
        # 如果长宽比不是1：1，还会强制把图片的比例修改回去
        # 意见：训练时一定要注意，千万别让这个干扰到了效果
        if a.aspect_ratio != 1.0:
            # upscale to correct aspect ratio
            size = [CROP_SIZE, int(round(CROP_SIZE * a.aspect_ratio))]
            image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)

        # tf.image.convert_image_dtype()函数的一些注意事项，该函数用于转换图像的像素值类型，
        # 参见：https://blog.csdn.net/cxx654/article/details/98373018，https://blog.csdn.net/wc781708249/article/details/78392754
        return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

    # 将input, output, target的像素值改变为uint8格式
    # reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("convert_inputs"):
        converted_inputs = convert(inputs)

    with tf.name_scope("convert_targets"):
        converted_targets = convert(targets)

    with tf.name_scope("convert_outputs"):
        converted_outputs = convert(outputs)

    # 创建了一个字典，储存了图片的路径，输入图片、输出图片、目标图片
    # tf.map_fn()，参见：https://blog.csdn.net/loseinvain/article/details/78815130
    with tf.name_scope("encode_images"):
        display_fetches = {
            "paths": examples.paths,
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
            "targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
        }

    # tf.summary.image()，似乎是与tensorboard有关，用于将结果可视化，
    # 参见：https://www.2cto.com/kf/201805/746214.html，https://www.cnblogs.com/lyc-seu/p/8647792.html
    # 参见：https://blog.csdn.net/zongza/article/details/89194559
    # 疑问：具体用法还有待进一步的调查、跑实例
    # summaries
    with tf.name_scope("inputs_summary"):
        tf.summary.image("inputs", converted_inputs)

    with tf.name_scope("targets_summary"):
        tf.summary.image("targets", converted_targets)

    with tf.name_scope("outputs_summary"):
        tf.summary.image("outputs", converted_outputs)

    with tf.name_scope("predict_real_summary"):
        tf.summary.image("predict_real", tf.image.convert_image_dtype(model.predict_real, dtype=tf.uint8))

    with tf.name_scope("predict_fake_summary"):
        tf.summary.image("predict_fake", tf.image.convert_image_dtype(model.predict_fake, dtype=tf.uint8))

    # tf.summary.scalar()也与结果可视化有关，参见：https://www.2cto.com/kf/201805/746214.html
    tf.summary.scalar("discriminator_loss", model.discrim_loss)
    tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)
    tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)

    # tf.summary.histogram()也与结果可视化有关，参见：https://www.2cto.com/kf/201805/746214.html
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)

    for grad, var in model.discrim_grads_and_vars + model.gen_grads_and_vars:
        tf.summary.histogram(var.op.name + "/gradients", grad)

    # tf.reduce_prod()函数，计算输入的tensor的所有元素的乘积，参见：https://blog.csdn.net/z2539329562/article/details/83506136
    # tf.reduce_sum()函数，计算输入的tensor的所有元素的和，参见：https://www.jianshu.com/p/30b40b504bae
    # 所以下列语句是计算可训练参数的总个数
    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    # tf.train.Saver()与check point有关，参见：https://blog.csdn.net/hl1hl/article/details/85638276
    # 注意：记得修改一下这个的参数，用以更好的使用checkpoint
    # 疑问：tf.train.Saver()与tf.train.Supervisor()的具体用法尚且有不明晰的地方，还有待进一步研究
    saver = tf.train.Saver(max_to_keep=1)  # max_to_keep表示要保留的最近文件的最大数量

    # a.trace_freq或a.summary_freq不为0时，载入输出路径
    logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None

    # tf.train.Supervisor()
    # 参见：https://www.jianshu.com/p/7490ebfa3de8
    # 疑问：tf.train.Saver()与tf.train.Supervisor()的具体用法尚且有不明晰的地方，还有待进一步研究
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    with sv.managed_session() as sess:
        print("parameter_count =", sess.run(parameter_count))  # 显示参数个数

        # 从最新的check point中恢复模型，参见：https://blog.csdn.net/sinat_30372583/article/details/79763044
        if a.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)  # 会自动找到最近保存的变量文件
            saver.restore(sess, checkpoint)

        # 设置max_steps
        max_steps = 2**32
        if a.max_epochs is not None:
            max_steps = examples.steps_per_epoch * a.max_epochs
        if a.max_steps is not None:
            max_steps = a.max_steps

        # 待解析……
        if a.mode == "test":
            # testing
            # at most, process the test data once
            start = time.time()
            max_steps = min(examples.steps_per_epoch, max_steps)
            for step in range(max_steps):
                results = sess.run(display_fetches)
                filesets = save_images(results)
                for i, f in enumerate(filesets):
                    print("evaluated image", f["name"])
                index_path = append_index(filesets)
            print("wrote index at", index_path)
            print("rate", (time.time() - start) / max_steps)
        else:
            # 模式=training

            # 返回当前时间戳，参见：https://www.runoob.com/python/att-time-time.html
            start = time.time()

            for step in range(max_steps):
                def should(freq):
                    # frep不为0，且满足第二条件：step+1能够整除frep，或者step到达最大值（step应该是从0开始算的）；返回真，否则假
                    return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

                options = None
                run_metadata = None
                if should(a.trace_freq):  # 满足trace条件
                    # 记录一些信息，参见：https://blog.csdn.net/qq_31120801/article/details/75268765
                    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                fetches = {
                    "train": model.train,
                    "global_step": sv.global_step,
                }

                if should(a.progress_freq):  # 满足progress条件
                    # 获取当前的各个loss的影子变量，将其添加到fetches中
                    fetches["discrim_loss"] = model.discrim_loss
                    fetches["gen_loss_GAN"] = model.gen_loss_GAN
                    fetches["gen_loss_L1"] = model.gen_loss_L1

                if should(a.summary_freq):  # 满足summary条件
                    # 疑问：暂时没有查到合适的解释，待解析。但大体意思上
                    fetches["summary"] = sv.summary_op

                if should(a.display_freq):  # 满足display条件
                    fetches["display"] = display_fetches

                # fetches中存储的是各种操作，这些操作通过sess.run()来运行
                # 疑问：这里的options与run_metadata的作用
                # 关于参数的作用，参见：https://blog.csdn.net/lllxxq141592654/article/details/89792885
                results = sess.run(fetches)

                if should(a.summary_freq):  # 满足summary条件
                    print("recording summary")
                    # 疑问：作用待解
                    sv.summary_writer.add_summary(results["summary"], results["global_step"])

                if should(a.display_freq):  # 满足display条件
                    print("saving display images")
                    # 疑问：作用待解
                    # 将目前正在训练的图片，以input, output, target一组保存下来
                    filesets = save_images(results["display"], step=results["global_step"])

                    # 将刚刚保存的图片名，以及对应的step，写入index.html文件
                    append_index(filesets, step=True)

                if should(a.trace_freq):  # 满足trace条件
                    print("recording trace")
                    # 参见：https://blog.csdn.net/dcrmg/article/details/79810142
                    # 疑问：代码效果不明，待进一步研究
                    sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

                if should(a.progress_freq):  # 满足progress条件
                    # global_step will have the correct step count if we resume from a checkpoint
                    train_epoch = math.ceil(results["global_step"] / examples.steps_per_epoch)  # 计算epoch数
                    train_step = (results["global_step"] - 1) % examples.steps_per_epoch + 1  # 计算当前epoch运行到了哪个stepp
                    rate = (step + 1) * a.batch_size / (time.time() - start)  # 从开始到现在，平均每秒处理多少图片
                    remaining = (max_steps - step) * a.batch_size / rate  # 预计还剩下多少每秒能够处理完
                    print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (train_epoch, train_step, rate, remaining / 60))
                    print("discrim_loss", results["discrim_loss"])
                    print("gen_loss_GAN", results["gen_loss_GAN"])
                    print("gen_loss_L1", results["gen_loss_L1"])

                if should(a.save_freq):  # 满足save条件
                    print("saving model")
                    # 疑问：关于下面是如何save的，尚且不明，待研究
                    saver.save(sess, os.path.join(a.output_dir, "model"), global_step=sv.global_step)

                # 疑问：关于什么时候才会触发下面的should_stop()，尚且不明。
                if sv.should_stop():
                    break
main()

