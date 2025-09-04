import cv2
import numpy as np
# 去噪
def denoise_nl_means(image, h=4, hForColor=4, templateWindowSize=5, searchWindowSize=19):
    denoised_image = cv2.fastNlMeansDenoisingColored(image, None, h, hForColor, templateWindowSize,
                                                     searchWindowSize)
    return denoised_image

#如果真的用改成class类减少复用
def image_deal(img_path):
    im=cv2.imread(img_path)
    im = denoise_nl_means(im)
    img_path=img_path.replace("steelpipe","steelpipe_enhance")
    cv2.imwrite(img_path, im)
if __name__== "__main__" :
    import os
    from pathlib import Path

    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[0]  # YOLO root directory
    path = ROOT /"../autodl-tmp/datasets/steelpipe/images/val2021"  # 文件夹目录
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    s = []
    NUM=0

    # image_deal("/root/Yolov9/../autodl-tmp/datasets/steelpipe/images/train2021/air-hole10-010.jpg")
    for file in files:  # 遍历文件夹
        print(NUM)
        file=os.path.join(path,file)
        image_deal(file)
        NUM=NUM+1
