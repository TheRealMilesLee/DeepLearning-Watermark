import os
import cv2 as cv
import glob

data_path = "/home/wangruowei/PycharmProjects/watermark2020/data/origin/test"
output_path = "/home/wangruowei/PycharmProjects/watermark2020/data/gray/test"

if not os.path.exists(output_path):
    os.mkdir(output_path)

data_name_list = glob.glob(os.path.join(data_path, "*.png"))

for i in range(len(data_name_list)):
    image_path = data_name_list[i]
    image_name = os.path.basename(image_path).split(".")[0]

    print(str(i) + ": " + image_name)

    image = cv.imread(image_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imwrite(os.path.join(output_path, image_name + ".png"), image)

