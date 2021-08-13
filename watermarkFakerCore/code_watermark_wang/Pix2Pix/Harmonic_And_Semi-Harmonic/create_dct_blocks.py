import math
import numpy as np
import tensorflow as tf

def alpha(x, k):
    if x == 0:
        return (1/k)**0.5
    else:
        return (2/k)**0.5

def one_dct_block(u, v, k):
    c_u_v = np.zeros([k, k])
    for m in range(k):
        for n in range(k):
            cos_part1 = math.cos((2*m+1) * math.pi * u / (2*k))
            cos_part2 = math.cos((2*n+1) * math.pi * v / (2*k))
            c_u_v[m, n] = alpha(u, k) * alpha(v, k) * cos_part1 * cos_part2
    return c_u_v

# k：represents the size of a harmonic kernel and the number of harmonic kernels
def create_dct_blocks(k):
    # [rows, columns, kernel_height, kernel_width]
    kernels_bank = np.zeros([k, k, k, k])
    for u in range(k):
        for v in range(k):
            dct_kernel = one_dct_block(u, v, k)
            kernels_bank[u, v] = dct_kernel
    return kernels_bank

# the function is designed for tensorflow.nn.conv2d
# 该函数针对tensorflow.nn.conv2d，为了产生正确的卷积核，对kernel的格式、顺序做了调整
def one_dct_block_tf(u, v, k):
    c_u_v = np.zeros([k, k, 1, 1])
    for m in range(k):
        for n in range(k):
            cos_part1 = math.cos((2*m+1) * math.pi * u / (2*k))
            cos_part2 = math.cos((2*n+1) * math.pi * v / (2*k))
            c_u_v[m, n, 0, 0] = alpha(u, k) * alpha(v, k) * cos_part1 * cos_part2
    return c_u_v

# k：represents the size of a harmonic kernel and the number of harmonic kernels
# the function is designed for tensorflow.nn.conv2d
# 该函数针对tensorflow.nn.conv2d，为了产生正确的卷积核，对kernel的格式、顺序做了调整
def create_dct_blocks_tf(k, isConv):
    kernels_bank = one_dct_block_tf(0, 0, k)
    for u in range(k):
        for v in range(k):
            if u == 0 and v == 0:
                continue
            dct_kernel = one_dct_block_tf(u, v, k)
            if isConv is True:
                kernels_bank = np.concatenate((kernels_bank, dct_kernel), axis=3)
            else:
                kernels_bank = np.concatenate((kernels_bank, dct_kernel), axis=2)
    # [kernel_height = k, kernel_width = k, in_channels = 1, out_channels = k*k]
    return kernels_bank


# create_dct_blocks_tf()函数的快速版本，尝试通过此改进提升网络结果的速度，2020年08月04日09:40:51，by王爇沩
def create_dct_blocks_tf_fast(k, isConv, input_channels_size):
    kernels_bank = one_dct_block_tf(0, 0, k)
    for u in range(k):
        for v in range(k):
            if u == 0 and v == 0:
                continue
            dct_kernel = one_dct_block_tf(u, v, k)
            if isConv is True:
                kernels_bank = np.concatenate((kernels_bank, dct_kernel), axis=3)
            else:
                kernels_bank = np.concatenate((kernels_bank, dct_kernel), axis=2)

    # now for convolution [kernel_height = k, kernel_width = k, in_channels = 1, out_channels = k*k]
    # now for de-convolution [kernel_height = k, kernel_width = k, out_channels = k*k, in_channels = 1]

    tmp = kernels_bank
    for i in range(1, input_channels_size):
        if isConv is True:
            tmp = np.concatenate([tmp, kernels_bank], axis=2)
        else:
            tmp = np.concatenate([tmp, kernels_bank], axis=3)
    kernels_bank = tmp
    # now for convolution [kernel_height = k, kernel_width = k, in_channels = n, out_channels = k*k]
    # now for de-convolution [kernel_height = k, kernel_width = k, out_channels = k*k, in_channels = n]

    tmp = kernels_bank
    for i in range(1, input_channels_size):
        if isConv is True:
            tmp = np.concatenate([tmp, kernels_bank], axis=3)
        else:
            tmp = np.concatenate([tmp, kernels_bank], axis=2)
    kernels_bank = tmp
    # now for convolution [kernel_height = k, kernel_width = k, in_channels = n, out_channels = n*k*k]
    # now for de-convolution [kernel_height = k, kernel_width = k, out_channels = n*k*k, in_channels = n]
    return kernels_bank