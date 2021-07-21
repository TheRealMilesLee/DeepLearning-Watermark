import cv2
import math
import numpy as np
from fractions import Fraction
theta = 0.5
h = 2
r11 = 1, r12 = 0, r13 = 0
r21 = 0, r22 = -math.cos(theta), r23 = -math.sin(theta)
r31 = 0, r32 = 0, r33 = 0
t_1 = 0, t_2 = fractions.Fraction(-h, math.tan(theta)), t_3 = h