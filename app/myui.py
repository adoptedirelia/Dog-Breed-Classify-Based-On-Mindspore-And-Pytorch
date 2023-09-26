# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'myui.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QTextEdit, QVBoxLayout, QWidget, QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QFont, QIcon
from PyQt5.QtCore import Qt
import utils
import mindspore
from mindspore import Tensor,context
import os
import numpy as np

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):

        data_folder = '../data/train_valid_test/train'

        self.breed_mapping = {}
        class_idx = 0
        file = os.listdir(data_folder)
        file = sorted(file)
        #print(file)
        for folder_name in file:
            if os.path.isdir(os.path.join(data_folder, folder_name)):
                self.breed_mapping[class_idx] = folder_name
                class_idx += 1


        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(420, 500)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.upload_image = QtWidgets.QPushButton(self.centralwidget)
        self.upload_image.setGeometry(QtCore.QRect(20, 400, 120, 50))
        self.upload_image.setObjectName("upload_image")
        self.load_model = QtWidgets.QPushButton(self.centralwidget)
        self.load_model.setGeometry(QtCore.QRect(20, 330, 120, 50))
        self.load_model.setObjectName("load_model")
        self.image = QtWidgets.QLabel(self.centralwidget)
        self.image.setGeometry(QtCore.QRect(10, 10, 400, 300))
        self.image.setAutoFillBackground(False)
        self.image.setText("")
        self.image.setObjectName("image")
        self.model_status = QtWidgets.QTextBrowser(self.centralwidget)
        self.model_status.setGeometry(QtCore.QRect(160, 340, 220, 30))
        self.model_status.setObjectName("model_status")
        self.dog_class = QtWidgets.QTextBrowser(self.centralwidget)
        self.dog_class.setGeometry(QtCore.QRect(160, 410, 220, 30))
        self.dog_class.setObjectName("dog_class")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 420, 24))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.mw = MainWindow
        tmp = '<span style="color: #FF5733;">模型未载入</span>'

        self.model_status.setHtml(tmp)
        self.model_status.setStyleSheet("background-color: transparent;")
        self.model = False
        self.net = utils.get_resnet(None)

        self.image.setScaledContents(True)


        self.load_model.clicked.connect(self.load_model_func)
        self.upload_image.clicked.connect(self.upload_image_func)
        #self.model_status.setText("你好")
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "🐶狗狗品种分类"))
        self.upload_image.setText(_translate("MainWindow", "上传图片"))
        self.load_model.setText(_translate("MainWindow", "载入模型"))

    def load_model_func(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        image_path, _ = QFileDialog.getOpenFileName(self.mw, "选择图片", "", "Models (*.ckpt *.pth);;All Files (*)", options=options)
        
        if image_path:

            param_dict = mindspore.load_checkpoint(image_path)
            mindspore.load_param_into_net(self.net, param_dict)
            self.model = True
            t = '<span style="color: #33FF00;">模型载入成功</span>'
            
            self.model_status.setHtml(t)


    def upload_image_func(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        if self.model == False:
            alert = QMessageBox()
            alert.setIcon(QMessageBox.Warning)
            alert.setWindowTitle("警告")
            alert.setText("请先载入模型")
            alert.setStandardButtons(QMessageBox.Ok)
            alert.exec_()
            return 
        image_path, _ = QFileDialog.getOpenFileName(self.mw, "选择图片", "", "Images (*.png *.jpg *.jpeg);;All Files (*)", options=options)
        
        if image_path:

            
            result = self.classify_variety(image_path)
            self.dog_class.setText(result)

            pixmap = QPixmap(image_path)
            self.image.setPixmap(pixmap)

    def classify_variety(self, image_path):
    



        preprocessed_image = utils.preprocess_single_image(image_path)


        # Convert the preprocessed image to a Tensor
        input_data = Tensor(preprocessed_image, dtype=mindspore.float32)

        # Run the model
        prediction = self.net(input_data)

        # Convert the prediction to a numpy array
        prediction = prediction.asnumpy()

        # Get the predicted class index
        predicted_class = np.argmax(prediction)



        # Get the predicted breed name
        predicted_breed = self.breed_mapping[predicted_class]
        #predicted_breed = predicted_class
        
        return predicted_breed