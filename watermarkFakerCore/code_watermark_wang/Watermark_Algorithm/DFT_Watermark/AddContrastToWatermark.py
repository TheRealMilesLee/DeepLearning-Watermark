import cv2 
import numpy as np
import os

SourcePath = "D:/CS-Related/Watermark Faker/TestFolder/Test_Images/DFT/Train images/Output/Reverse/5_de.png"
SavePath = "D:/CS-Related/Watermark Faker/TestFolder/Test_Images/DFT/Train images/Output/Reverse/Contrast"

img = cv2.imread(SourcePath,0)
# 增加图像亮度
# 注意需要使用cv.add(),不能直接x+y
res1 = np.uint8(np.clip((cv2.add(1.8*img,30)), 0, 255))
# 增加图像对比度
res2 = np.uint8(np.clip((cv2.add(1.5*img,0)), 0, 255))

save_path = os.path.join(SavePath, "_Contrast.png")

cv2.imwrite(save_path, res1)