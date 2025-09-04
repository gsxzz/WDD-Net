import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

    def gamma(self,image):
        fgamma = 2
        image_gamma = np.uint8(np.power((np.array(image) / 255.0), fgamma) * 255.0)
        cv2.normalize(image_gamma, image_gamma, 0, 255, cv2.NORM_MINMAX)
        cv2.convertScaleAbs(image_gamma, image_gamma)
        return image_gamma

def image_deal(img_path,img_size=640,augment=False):
    im = cv2.imread(img_path)
    #去噪
    im = cv2.medianBlur(im, 3)  # 中值滤波函数
    # im=self.gamma(medBlur)
    img = np.uint8(im)
    imgr = img[:, :, 0]
    imgg = img[:, :, 1]
    imgb = img[:, :, 2]

    claher = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 5))
    claheg = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 5))
    claheb = cv2.createCLAHE(clipLimit=1, tileGridSize=(8, 5))
    cllr = claher.apply(imgr)
    cllg = claheg.apply(imgg)
    cllb = claheb.apply(imgb)

    im = np.dstack((cllr, cllg, cllb))


    h0, w0 = im.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # ratio
    if r != 1:  # if sizes are not equal
        interp = cv2.INTER_LINEAR if (augment or r > 1) else cv2.INTER_AREA
        im = cv2.resize(im, (int(w0 * r), int(h0 * r)), interpolation=interp)
    im=letterbox(im)[0]
    # im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous
    print(im.shape)
    img_path=img_path.replace("steelpipe","steelpipe_moco")
    cv2.imwrite(img_path, im)
if __name__ == "__main__":
    import os
    from pathlib import Path

    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[0]  # YOLO root directory
    path = ROOT / "../../autodl-tmp/datasets/steelpipe/images/train2021"  # 文件夹目录
    print(path)
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    s = []
    NUM = 0

    # image_deal("/root/Yolov9/../autodl-tmp/datasets/steelpipe/images/train2021/air-hole10-010.jpg")
    for file in files:  # 遍历文件夹
        print(NUM)
        file = os.path.join(path, file)

        img=image_deal(file)
        NUM = NUM + 1
