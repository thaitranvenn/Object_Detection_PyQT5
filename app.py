import sys
import cv2
import argparse
import random
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from PyQt5 import QtCore, QtGui, QtWidgets

from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.datasets import letterbox
from utils.plots import plot_one_box

class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)
        self.timerVideo = QtCore.QTimer()
        self.setupUi(self)
        self.initLogo()
        self.initSlots()
        self.cap = cv2.VideoCapture()
        self.out = None
        # self.out = cv2.VideoWriter('prediction.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640, 480))

        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str,
                            default='weights/yolov5n.pt', help='model.pt path(s)')
        # file/folder, 0 for webcam
        parser.add_argument('--source', type=str,
                            default='data/images', help='source')
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

        source, weights, view_img, save_txt, imgsz = self.opt.source, self.opt.weights, self.opt.view_img, self.opt.save_txt, self.opt.img_size

        self.device = select_device(self.opt.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        cudnn.benchmark = True

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

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(831, 537)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout.setContentsMargins(-1, -1, 0, -1)
        self.verticalLayout.setSpacing(80)
        self.verticalLayout.setObjectName("verticalLayout")
        self.pushButton_Image = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_Image.sizePolicy().hasHeightForWidth())
        self.pushButton_Image.setSizePolicy(sizePolicy)
        self.pushButton_Image.setMinimumSize(QtCore.QSize(150, 100))
        self.pushButton_Image.setMaximumSize(QtCore.QSize(150, 100))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(12)
        self.pushButton_Image.setFont(font)
        self.pushButton_Image.setObjectName("pushButton_Image")
        self.verticalLayout.addWidget(self.pushButton_Image)
        self.pushButton_Camera = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_Camera.sizePolicy().hasHeightForWidth())
        self.pushButton_Camera.setSizePolicy(sizePolicy)
        self.pushButton_Camera.setMinimumSize(QtCore.QSize(150, 100))
        self.pushButton_Camera.setMaximumSize(QtCore.QSize(150, 100))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(12)
        self.pushButton_Camera.setFont(font)
        self.pushButton_Camera.setObjectName("pushButton_Camera")
        self.verticalLayout.addWidget(self.pushButton_Camera)
        self.pushButton_Video = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_Video.sizePolicy().hasHeightForWidth())
        self.pushButton_Video.setSizePolicy(sizePolicy)
        self.pushButton_Video.setMinimumSize(QtCore.QSize(150, 100))
        self.pushButton_Video.setMaximumSize(QtCore.QSize(150, 100))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(12)
        self.pushButton_Video.setFont(font)
        self.pushButton_Video.setObjectName("pushButton_Video")
        self.verticalLayout.addWidget(self.pushButton_Video)
        self.verticalLayout.setStretch(2, 1)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.horizontalLayout.setStretch(1, 3)
        self.horizontalLayout_2.addLayout(self.horizontalLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 831, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Objects Detection using YOLOV5 and PyQT5"))
        self.pushButton_Image.setText(_translate("MainWindow", "Open Image"))
        self.pushButton_Camera.setText(_translate("MainWindow", "Open Camera"))
        self.pushButton_Video.setText(_translate("MainWindow", "Open Video"))
        self.label.setText(_translate("MainWindow", "TextLabel"))

    def initSlots(self):
        self.pushButton_Image.clicked.connect(self.buttonOpenImage)
        self.pushButton_Video.clicked.connect(self.buttonOpenVideo)
        self.pushButton_Camera.clicked.connect(self.buttonOpenCamera)
        self.timerVideo.timeout.connect(self.showVideoFrame)

    def initLogo(self):
        pix = QtGui.QPixmap('github.png')
        self.label.setScaledContents(True)
        self.label.setPixmap(pix)

    def buttonOpenImage(self):
        print('Opening a image!\nPlease choose one!')
        name_list = []

        img_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Image", "", "*.jpg;;*.png;;All Files(*)")
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

    def buttonOpenVideo(self):
        videoName, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Video", "", "*.mp4;;*.avi;;All Files(*)")

        if not videoName:
            return

        if not self.timerVideo.isActive():
            flag = self.cap.open(videoName)
            if flag == False:
                QtWidgets.QMessageBox.warning(
                    self, u"Warning", u"Failed to open video", buttons=QtWidgets.QMessageBox.Ok, defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.out = cv2.VideoWriter('result_Video.avi', cv2.VideoWriter_fourcc(
                    *'MJPG'), 20, (int(self.cap.get(3)), int(self.cap.get(4))))
                self.timerVideo.start(30)
                self.pushButton_Image.setDisabled(True)
                self.pushButton_Video.setText(u"Turn off the video")
                self.pushButton_Camera.setDisabled(True)
        else:
            self.timerVideo.stop()
            self.cap.release()
            self.out.release()
            self.label.clear()
            self.initLogo()
            self.pushButton_Image.setDisabled(False)
            self.pushButton_Video.setText(u"Video Detection")
            self.pushButton_Camera.setDisabled(False)

    def buttonOpenCamera(self):
        if not self.timerVideo.isActive():
            # Use local camera by default
            flag = self.cap.open(0)
            if flag == False:
                QtWidgets.QMessageBox.warning(
                    self, u"Warning", u"Failed to open camera", buttons=QtWidgets.QMessageBox.Ok, defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.out = cv2.VideoWriter('result_Camera.avi', cv2.VideoWriter_fourcc(
                    *'MJPG'), 20, (int(self.cap.get(3)), int(self.cap.get(4))))
                self.timerVideo.start(30)
                self.pushButton_Image.setDisabled(True)
                self.pushButton_Video.setDisabled(True)
                self.pushButton_Camera.setText(u"Turn off the camera")
        else:
            self.timerVideo.stop()
            self.cap.release()
            self.out.release()
            self.label.clear()
            self.initLogo()
            self.pushButton_Video.setDisabled(False)
            self.pushButton_Image.setDisabled(False)
            self.pushButton_Camera.setText(u"Camera detection")

    def showVideoFrame(self):
        name_list = []

        flag, image = self.cap.read()
        if image is not None:
            showImage = image
            with torch.no_grad():
                image = letterbox(image, new_shape=self.opt.img_size)[0]
                # Convert
                # BGR to RGB, to 3x416x416
                image = image[:, :, ::-1].transpose(2, 0, 1)
                image = np.ascontiguousarray(image)
                image = torch.from_numpy(image).to(self.device)
                image = image.half() if self.half else image.float()  # uint8 to fp16/32
                image /= 255.0  # 0 - 255 to 0.0 - 1.0
                if image.ndimension() == 3:
                    image = image.unsqueeze(0)
                # Inference
                prediction = self.model(image, augment=self.opt.augment)[0]

                # Apply NMS
                prediction = non_max_suppression(prediction, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                           agnostic=self.opt.agnostic_nms)
                # Process detections
                for i, det in enumerate(prediction):  # detections per image
                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(
                            image.shape[2:], det[:, :4], showImage.shape).round()
                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            label = '%s %.2f' % (self.names[int(cls)], conf)
                            name_list.append(self.names[int(cls)])
                            print(label)
                            plot_one_box(
                                xyxy, showImage, label=label, color=self.colors[int(cls)], line_thickness=2)

            self.out.write(showImage)
            show = cv2.resize(showImage, (640, 480))
            self.result = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                     QtGui.QImage.Format_RGB888)
            self.label.setPixmap(QtGui.QPixmap.fromImage(showImage))

        else:
            self.timerVideo.stop()
            self.cap.release()
            self.out.release()
            self.label.clear()
            self.pushButton_Video.setDisabled(False)
            self.pushButton_Image.setDisabled(False)
            self.pushButton_Camera.setDisabled(False)
            self.initLogo()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    ui.show()
    sys.exit(app.exec_())