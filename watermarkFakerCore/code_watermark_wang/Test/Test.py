#import cv2
import math
import numpy as np
from fractions import Fraction
theta = 0.5
h = 2
r11 = 1, r12 = 0, r13 = 0
r21 = 0, r22 = -math.cos(theta), r23 = -math.sin(theta)
r31 = 0, r32 = 0, r33 = 0
t_1 = 0, t_2 = Fraction(-h,math.tan(theta)), t_3 = h
'''
a = np.array([[f_x, 0, c_x],
             [0, f_y, c_y],
             [0, 0, 1]])

b = np.array([[r11, r12, r13, t_1],
             [r21, r22, r23, t_2],
             [r31, r32, r33, t_3]])

c = np.array([[X],
              [Y],
              [Z],
              [1]])
'''









