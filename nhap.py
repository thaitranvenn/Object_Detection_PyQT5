from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox
import sys
import cv2
import numpy as np
import torch
import argparse
import random
import torch.backends.cudnn as cudnn
from PyQt5.uic import loadUi

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.datasets import letterbox
from utils.torch_utils import select_device
from utils.plots import plot_one_box

from os import environ

def suppress_qt_warnings():
    environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    environ["QT_SCALE_FACTOR"] = "1"

suppress_qt_warnings()

class LoadQt(QMainWindow):
    def __init__(self, parent=None):
        super(LoadQt, self).__init__(parent)
        self.timerVideo = QtCore.QTimer()
        loadUi('nhap.ui',self)

        self.cap = cv2.VideoCapture()
        self.out = None
        self.setWindowIcon(QtGui.QIcon("python-icon.png"))

        # Set default picture for PyQt5
        picture = QtGui.QPixmap('github.png')
        self.label.setScaledContents(True)
        self.label.setPixmap(picture)

        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str,
                            default='weights/yolov5x.pt')
        # file/folder, 0 for webcam
        parser.add_argument('--source', type=str,
                            default='data/images')
        parser.add_argument('--img-size', type=int,
                            default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float,
                            default=0.25, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float,
                            default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--device', default='',
                            help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument(
            '--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', action='store_true',
                            help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true',
                            help='save confidences in --save-txt labels')
        parser.add_argument('--nosave', action='store_true',
                            help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int,
                            help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument(
            '--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true',
                            help='augmented inference')
        parser.add_argument('--update', action='store_true',
                            help='update all models')
        parser.add_argument('--project', default='runs/detect',
                            help='save results to project/name')
        parser.add_argument('--name', default='exp',
                            help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true',
                            help='existing project/name ok, do not increment')
        self.opt = parser.parse_args()
        print(self.opt)

        weights, imgsz = self.opt.weights, self.opt.img_size
        
        self.device = select_device(self.opt.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        cudnn.benchmark = True # set True to speed up constant image size inference

        # Load model
        self.model = attempt_load(
            weights, map_location=self.device)  # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if self.half:
            self.model.half()  # to FP16

        # Get names and colors
        self.names = self.model.module.names if hasattr(
            self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255)
                        for _ in range(3)] for _ in self.names]

        # actions menu
        self.actionExit.triggered.connect(self.questionMessage)
        self.actionAbout_QT.triggered.connect(self.aboutMessage)
        self.actionAuthor.triggered.connect(self.aboutMessage2)

        # buttons menu
        self.pushButton_Image.clicked.connect(self.buttonOpenImage)

    @pyqtSlot()
    def buttonOpenImage(self):
        print('Opening a image!\nPlease choose one!')
        name_list = []

        img_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Image", "", "All Files(*);;*.jpg;;*.png")
        if not img_name:
            return

        img = cv2.imread(img_name)
        print(img_name)
        showimg = img
        with torch.no_grad():
            img = letterbox(img, new_shape=self.opt.img_size)[0]
            # Convert
            # BGR to RGB, to 3x416x416
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            # Inference
            prediction = self.model(img, augment=self.opt.augment)[0]
            # Apply NMS
            prediction = non_max_suppression(prediction, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                       agnostic=self.opt.agnostic_nms)
            print(prediction)
            # Process detections
            for i, det in enumerate(prediction):
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], showimg.shape).round()

                    for *xyxy, conf, cls in reversed(det):
                        label = '%s %.2f' % (self.names[int(cls)], conf)
                        name_list.append(self.names[int(cls)])
                        plot_one_box(xyxy, showimg, label=label,
                                     color=self.colors[int(cls)], line_thickness=2)

        cv2.imwrite('result_Image.jpg', showimg)
        self.result = cv2.cvtColor(showimg, cv2.COLOR_BGR2BGRA)
        self.result = cv2.resize(
            self.result, (640, 480), interpolation=cv2.INTER_AREA)
        self.QtImg = QtGui.QImage(
            self.result.data, self.result.shape[1], self.result.shape[0], QtGui.QImage.Format_RGB32)
        self.label.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))

    def aboutMessage(self):
        QMessageBox.about(self, "About Qt - Qt Designer",
            "Qt is a cross-platform framework written in C++ language, used to develop projects, software for desktop, or mobile."
            "PyQt is a library that allows you to use Qt GUI, a very famous framework of C++. PyQt has many versions but the most recent and most supported is PyQt5.\n"
        )
    def aboutMessage2(self):
        QMessageBox.about(self, "About Author", "Executor: Tran Huu Thai\n\n"
                                                    "Contact me: \n"
                                                    "Number Phone - 89138032486\n"
                                                    "Email - Thaitran130399.tusur@gmail.com"
                          )

    def questionMessage(self):
        message = QMessageBox.question(self, "Exit", "Do you want to exit? Please sure that you saved all your data!", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if message == QMessageBox.Yes:
            print("Yes")
            self.close()
        else:
            print("No")

application = QApplication(sys.argv)
windows = LoadQt()
windows.show()
sys.exit(application.exec())