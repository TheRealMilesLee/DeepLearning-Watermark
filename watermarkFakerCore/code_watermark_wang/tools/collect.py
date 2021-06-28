import os
import cv2 as cv
import glob

data_path = "/home/wangruowei/桌面/data_collections/Caltech-256/data/256_ObjectCategories"
output_path = "/home/wangruowei/桌面/PycharmProjects/watermark2020/data/origin"

data_name_list = glob.glob(os.path.join(data_path, "*/*.jpg"))

for i in range(len(data_name_list)):
    image_path = data_name_list[i]
    image_name = os.path.basename(image_path).split(".")[0]

    print(str(i) + ": " + image_name)

    image = cv.imread(image_path)
    cv.imwrite(os.path.join(output_path, image_name + ".png"), image)

