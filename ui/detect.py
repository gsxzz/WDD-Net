import sys

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QDir
# from detect.detect_face.model import hsv_detect, haar, cnn


class MyWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.file_name = ''
        # 设置窗口标题和大小
        self.setWindowTitle(' ')
        self.setGeometry(100, 100, 1280, 1000)

        self.label = QtWidgets.QLabel()
        self.label.setGeometry(20, 20, 160, 60)  # 设置按钮的位置和大小

        # 上传文件按钮
        button = QPushButton('上传文件', self)
        button.setGeometry(20, 20, 160, 60)  # 设置按钮的位置和大小
        button.clicked.connect(self.select_image)  # 连接按钮的点击事件

        # 检测图像内容按钮
        button = QPushButton('HAAR检测', self)
        button.setGeometry(20, 100, 160, 60)  # 设置按钮的位置和大小
        button.clicked.connect(self.image_haar_deal)  # 连接按钮的点击事件

        # 检测图像内容按钮
        button = QPushButton('肤色检测', self)
        button.setGeometry(20, 180, 160, 60)  # 设置按钮的位置和大小
        button.clicked.connect(self.image_hsv_deal)  # 连接按钮的点击事件

        # 检测图像内容按钮
        button = QPushButton('深度学习检测', self)
        button.setGeometry(20, 260, 160, 60)  # 设置按钮的位置和大小
        button.clicked.connect(self.image_cnn_deal)  # 连接按钮的点击事件

        # 在窗口中添加一个标签用于显示图像
        self.image_label = QLabel(self)
        self.image_label.setGeometry(200, 20, 740, 500)

    def select_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        desktop_path = QDir.currentPath()
        self.file_name, _ = QFileDialog.getOpenFileName(self, "选择图像资源", desktop_path,
                                                        "Images (*.png *.xpm *.jpg *.bmp *.gif)", options=options)
        if self.file_name:
            pixmap = QPixmap(self.file_name)
            self.image_label.setPixmap(pixmap)
            self.image_label.setScaledContents(True)  # 让图像自适应 QLabel 大小
            self.image_label.show()

    # 设置frame的背景图片
    def set_frame_background(self, background_img):
        pix = background_img.convert("RGBA").toqpixmap()
        pix = pix.scaled(self.image_label.width(), self.image_label.height())
        self.image_label.setPixmap(pix)

    # harr分类器
    def image_haar_deal(self):
        try:
            if self.file_name != None:
                self.set_frame_background(haar(self.file_name))
        except Exception as e:
            print(e)

    # 肤色特征
    def image_hsv_deal(self):
        if self.file_name != None:
            self.set_frame_background(hsv_detect(self.file_name))

    # 肤色特征
    def image_cnn_deal(self):
        if self.file_name != None:
            self.set_frame_background(cnn(self.file_name))


if __name__ == '__main__':
    # 创建应用程序对象
    app = QApplication(sys.argv)
    # 创建窗口实例
    window = MyWindow()
    # 显示窗口
    window.show()
    # 运行应用程序的主循环
    sys.exit(app.exec_())