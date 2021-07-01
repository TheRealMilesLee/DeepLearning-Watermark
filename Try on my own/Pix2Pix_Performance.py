import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math
import time
import csv


#从硬盘中读取已经预处理完成的CSV文件
def read_module_from_disk():
    PreprocessLocation = ""
    
    PreprocessPath = tf.keras.utils.get_file("Preprocess.csv", PreprocessLocation)
    np.set_printoptions(precision = 3, suppress = True)

