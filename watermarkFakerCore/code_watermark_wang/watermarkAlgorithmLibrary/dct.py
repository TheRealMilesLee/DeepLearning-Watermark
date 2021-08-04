import cv2
import numpy as np
from watermarkAlgorithmLibrary import BaseWatermark

class DCT(BaseWatermark):
  def __init__(self, block_size = 8, save_watermark = False):
    BaseWatermark.__init__(self, save_watermark)
    self.block_size = block_size
  
  # 水印嵌入函数
  def embed(self, image, watermark):
    # 获得图片的长和宽
    h1, w1 = image.shape
    # 把原图片转换为float32格式
    image = image.astype('float32')
    # 获取转换后的图片的长和宽数据
    h2, w2 = watermark.shape
    # 将图片归化到0，1区间
    watermark = np.round(watermark / 255)  # [0, 255] -> {0, 1}; binary image
    
    if len(image.shape) != 2:
      raise TypeError("Image to embed DFT should be grayscale")
    if h1 != h2 * self.block_size and w1 != w2 * self.block_size:
      raise ValueError("Watermark's shape should be the '1/%d' of image's shape" % self.block_size)
    if self.block_size != 8:
      raise NotImplementedError("Block size should be 8; the other values version maybe implemented in the future")
    
    # 水印图片保存
    if self.save_watermark:
      cv2.imwrite('./images/Watermark_DCT_Method.png', (watermark * 255).astype('uint8'))
    
    B = self.block_size  # temporary value
    # 把图片周围填充一圈0
    image_wm = np.zeros(image.shape)
    for heightLoop in range(h2):
      for widthLoop in range(w2):
        # 针对图片的长和宽嵌入水印
        sub_image = image[heightLoop * B: (heightLoop + 1) * B, widthLoop * B: (widthLoop + 1) * B]
        sub_image_dct = cv2.dct(sub_image)
        if watermark[heightLoop, widthLoop] == 0:
          if sub_image_dct[3, 3] > sub_image_dct[4, 4]:
            temp = sub_image_dct[3, 3]
            sub_image_dct[3, 3] = sub_image_dct[4, 4]
            sub_image_dct[4, 4] = temp
        else:
          if sub_image_dct[3, 3] < sub_image_dct[4, 4]:
            temp = sub_image_dct[3, 3]
            sub_image_dct[3, 3] = sub_image_dct[4, 4]
            sub_image_dct[4, 4] = temp
        image_wm[heightLoop * B: (heightLoop + 1) * B, widthLoop * B: (widthLoop + 1) * B] = cv2.idct(sub_image_dct)
    return image_wm.astype('uint8')
  
  def extract(self, image_wm, image = None):
    h1, w1 = image_wm.shape
    B = self.block_size
    h2, w2 = h1 // B, w1 // B
    watermark_ = np.zeros((h2, w2))
    for i in range(h2):
      for j in range(w2):
        sub_image_wm = image_wm[i * B: (i + 1) * B, j * B: (j + 1) * B].astype('float32')
        sub_image_wm_dct = cv2.dct(sub_image_wm)
        if sub_image_wm_dct[3, 3] < sub_image_wm_dct[4, 4]:
          watermark_[i, j] = 0
        else:
          watermark_[i, j] = 1
    return (watermark_ * 255).astype('uint8')
