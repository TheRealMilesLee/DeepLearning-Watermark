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
    TRAIN_DATA_LOACTION = ""
    TEST_DATA_LOCATION = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"
    
    train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
    test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)
    np.set_printoptions(precision = 3, suppress = True)

