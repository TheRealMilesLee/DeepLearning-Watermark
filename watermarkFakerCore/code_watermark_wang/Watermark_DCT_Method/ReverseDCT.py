import os
import cv2
from watermarkAlgorithmLibrary import dct

if __name__ == '__main__':
    algorithm = dct.DCT
    origin_path = "D:/CS-Related/Watermark Faker/TestFolder/Test_Images/DCT/images/Source"  # original images
    source_path = "D:/CS-Related/Watermark Faker/TestFolder/Test_Images/DCT/images/Output"  # watermarked images
    output_path = "D:/CS-Related/Watermark Faker/TestFolder/Test_Images/DCT/images/Reverse"  # extracted watermark
    RGB = 0

    alg = algorithm()
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    i = 0
    for file_name in os.listdir(source_path):
        image_wm_path = os.path.join(source_path, file_name)
        origin_name = file_name.split("_")[0] + ".png"
        image_path = os.path.join(origin_path, origin_name)

        image_wm = cv2.imread(image_wm_path, flags=RGB)
        image = cv2.imread(image_path, flags=RGB)

        extracted_watermark_path = os.path.join(output_path, file_name.split(".")[0] + "_extract.png")
        extracted_watermark = alg.extract(image_wm, image)
        cv2.imwrite(extracted_watermark_path, extracted_watermark)

        i += 1
        print(i)

    print("finished")
