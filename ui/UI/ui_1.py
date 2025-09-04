from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PIL.ImageQt import ImageQt
from PyQt5.QtGui import QPalette, QPixmap, QBrush
from PyQt5.QtWidgets import QLabel, QHBoxLayout, QPushButton, QRadioButton, QVBoxLayout, QFileDialog, \
    QFrame, QDialog, QComboBox
import numpy as np
import sys
import cv2

from ui.UI.qss import main_btn_qss


class Ui_MainWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)  # 父类的构造函数

        self.cap = None
        self.thread_video = None
        self.thread = None
        self.imgName = None
        self.label_show_camera = None
        self.frame = None
        self.choose_btn_division = None
        self.choose_btn_detect = None
        self.video_re_lab = None
        self.picture_re_lab = None
        self.__layout_main = None
        self.image = None
        self.resize(800, 300)
        self.geometry()
        self.timer_camera = QTimer()  # 定义定时器
        self.dialog = QDialog()
        self.set_ui()  # 初始化程序界面

    def set_ui(self):
        self.setWindowTitle('钢结构焊缝缺陷智能识别系统')
        self.__layout_main = QVBoxLayout()  # 总布局
        # 页面组件
        title_lab = QLabel("焊缝缺陷图像增强")
        title_lab.setAlignment(Qt.AlignCenter)
        self.picture_re_lab = QPushButton("焊缝缺陷图像增强")
        self.video_re_lab = QPushButton("焊缝缺陷检测")

        self.choose_btn_detect = QRadioButton("口罩检测")
        self.choose_btn_division = QRadioButton("口罩分割")

        self.label_show_camera = QtWidgets.QLabel()  # 定义显示视频的Label
        self.label_show_camera.setFixedSize(480, 300)
        self.set_background(r"D:\bjut\pyqt5\ui\UI\background.jpeg")

        h_lay = QHBoxLayout()
        h_lay.addWidget(title_lab)

        h_lay_1=QHBoxLayout()
        h_lay_low = QVBoxLayout()
        h_lay_low.addWidget(self.picture_re_lab, 5)
        h_lay_low.addWidget(self.video_re_lab, 5)

        h_lay_1.addStretch(1)
        h_lay_1.addLayout(h_lay_low,3)
        h_lay_1.addStretch(1)

        img_show = QHBoxLayout()
        img_show.addWidget(self.label_show_camera)

        img_show_1 = QHBoxLayout()
        img_show_1.addWidget(self.label_show_camera)

        self.__layout_main.addLayout(img_show,2)
        self.__layout_main.addLayout(h_lay_1, 5)
        self.__layout_main.addLayout(img_show_1,2)

        self.setStyleSheet(main_btn_qss)
        self.setLayout(self.__layout_main)
        # 事件绑定
        self.picture_re_lab.clicked.connect(self.open_picture)
        # self.choose_btn_detect.setChecked(True)
        self.timer_camera.timeout.connect(self.show_camera)
        # self.video_re_lab.clicked.connect(self.play_video)
        self.video_re_lab.clicked.connect(self.open_picture)


    # 设置背景图片
    def set_background(self, background_img_path):
        pal = QPalette(self.palette())
        pix = QPixmap(background_img_path)
        pix = pix.scaled(self.width(), self.height())
        pal.setBrush(self.backgroundRole(), QBrush(pix))
        self.setAutoFillBackground(True)
        self.setPalette(pal)

    # 设置frame的背景图片
    def set_frame_background(self, background_img):
        pix = ImageQt(background_img)
        pix = pix.scaled(self.label_show_camera.width(), self.label_show_camera.height())
        self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(pix))

    def show_dialog(self):
        self.dialog.resize(260, 150)
        label = QLabel("请等待检测完成")
        label.setStyleSheet("QLabel{font:40px;font-family:songti;}")
        dialog_layout = QVBoxLayout()
        dialog_layout.addWidget(label)
        self.dialog.setLayout(dialog_layout)
        self.dialog.setWindowTitle("进行中")
        self.dialog.setWindowModality(Qt.ApplicationModal)
        self.dialog.exec_()

    def judge_which(self):
        if self.choose_btn_detect.isChecked():
            id = 0
        else:
            id = 1
        return id

    def open_picture(self):
        # id = self.judge_which()
        self.imgName = QFileDialog.getOpenFileName(self, "上传图片", "", "*.jpg *.png *.jpeg *.bmp")
        # if self.imgName[0] != "":
        #     if id == 0:
        #         self.thread = RunThread(self.imgName[0])
        #         self.thread.model_use = False
        #         self.thread.sin.connect(self.stop_dialog)
        #         self.thread.start()  # 启动线程
        #         self.show_dialog()
        #         background_img = self.thread.background_img
        #         # print(background_img, "预测图片")
        #         self.set_frame_background(background_img)
        #     else:
        #         self.thread = RunThread(self.imgName[0])
        #         self.thread.model_use = True
        #         self.thread.sin.connect(self.stop_dialog)
        #         self.thread.start()  # 启动线程
        #         self.show_dialog()
        #         background_img = self.thread.background_img
        #         print(background_img, "预测图片")
        #         self.set_frame_background(background_img)
        self.set_frame_background(self.imgName[0])

    def play_video(self):
        id = self.judge_which()
        self.imgName = QFileDialog.getOpenFileName(self, "上传视频", "", "*.mp4 *.avi")
        if self.imgName[0] != "":
            if id == 0:
                self.thread_video = RunThread_Video(self.imgName[0])
                self.thread_video.model_use = False
                self.thread_video.sin.connect(self.stop_dialog)
                self.thread_video.start()  # 启动线程
                self.show_dialog()
                self.division_model_video(r"/Users/wangziwei/PycharmProjects/graduate/UI/res.mp4")
            else:
                self.thread_video = RunThread_Video(self.imgName[0])
                self.thread_video.model_use = True
                self.thread_video.sin.connect(self.stop_dialog)
                self.thread_video.start()  # 启动线程
                self.show_dialog()
                self.division_model_video(r"/Users/wangziwei/PycharmProjects/graduate/UI/res.mp4")

    def division_model_video(self, video):
        try:
            self.cap = cv2.VideoCapture(video)  # 视频流
            self.timer_camera.start(100)
            self.timer_camera.timeout.connect(self.show_camera)
        except Exception as e:
            print(e)

    def show_camera(self):
        flag, self.image = self.cap.read()  # 从视频流中读取
        if flag:
            show = cv2.resize(self.image, (
                self.label_show_camera.width(), self.label_show_camera.height()))  # 把读到的帧的大小重新设置为 640x480
            show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)  # 视频色彩转换回RGB，这样才是现实的颜色
            show_image = QtGui.QImage(show.data, show.shape[1], show.shape[0],
                                      QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
            self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(show_image))  # 往显示视频的Label里 显示QImage
        else:
            self.cap.release()
            self.timer_camera.stop()  # 停止计时器

    def stop_dialog(self):
        self.dialog.close()


class RunThread(QThread):
    sin = pyqtSignal()

    def __init__(self, img_path):
        self.img_path = img_path
        self.model_use = True
        self.background_img = None
        super(RunThread, self).__init__()




class RunThread_Video(QThread):  # 要修改
    sin = pyqtSignal()

    def __init__(self, video_path):
        self.video = None
        self.fourcc = None
        self.model_use = True
        self.cap = cv2.VideoCapture(video_path)  # 视频流
        super(RunThread_Video, self).__init__()

    def run(self):
        self.show_camera()
        self.sin.emit()
        print("信号发送成功")




if __name__ == '__main__':
    import os

    os.environ['QT_MAC_WANTS_LAYER'] = '1'
    app = QtWidgets.QApplication(sys.argv)  # 固定的，表示程序应用
    ui = Ui_MainWindow()  # 实例化Ui_MainWindow
    ui.show()  # 调用ui的show()以显示。同样show()是源于父类QtWidgets.QWidget的
    # print("12312431234")
    sys.exit(app.exec_())  # 不加这句，程序界面会一闪而过
