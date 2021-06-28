import os
import shutil
import glob

input_path = "/home/wangruowei/PycharmProjects/watermark2020/data/pair_dct/train"
output_path = "/home/wangruowei/PycharmProjects/watermark2020/data/pair_dct"

files = glob.glob(os.path.join(input_path, "*.png"))

i = 0
for file in files:
    try:
        shutil.move(file, output_path)
    except:
        continue

    print(i)
    i += 1

print("finished")
