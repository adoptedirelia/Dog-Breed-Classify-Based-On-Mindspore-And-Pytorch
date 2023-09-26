import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QTextEdit, QVBoxLayout, QWidget, QFileDialog
from PyQt5.QtGui import QPixmap, QFont, QIcon
from PyQt5.QtCore import Qt
import os
import utils
import mindspore
from mindspore import Tensor,context
import os
import numpy as np


class VarietyClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("🐶狗狗品种分类")
        self.setFixedSize(420, 450)  # 固定窗口大小
        self.setWindowIcon(QIcon("./Irelia.png"))  # 请将 icon.png 替换为你的图标文件路径

        layout = QVBoxLayout()

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(400, 300)  # 固定 QLabel 大小
        self.image_label.setScaledContents(True)  # 图像自适应
        layout.addWidget(self.image_label)

        self.upload_button = QPushButton("上传图片", self)
        self.upload_button.clicked.connect(self.upload_image)
        layout.addWidget(self.upload_button)

        self.result_text = QTextEdit(self)
        self.result_text.setFixedSize(400, 50)  # 固定输出框大小
        self.result_text.setReadOnly(True)
        layout.addWidget(self.result_text)

        central_widget = QWidget(self)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.setStyleSheet(
            "QMainWindow { background-color: #f7f7f7; }"
            "QLabel { background-color: white; border: 2px solid #ddd; border-radius: 12px; padding: 30px; }"
            "QPushButton { background-color: #4CAF50; color: white; border: none; border-radius: 12px; padding: 15px 30px; }"
            "QPushButton:hover { background-color: #45a049; }"
            "QTextEdit { background-color: white; border: 2px solid #ddd; border-radius: 12px; padding: 15px; }"
        )

        self.result_text.setFont(QFont("Arial", 14))
        self.result_text.setStyleSheet("QTextEdit:focus { border: 2px solid #4CAF50; }")

    def upload_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        image_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg);;All Files (*)", options=options)
        
        if image_path:
            result = self.classify_variety(image_path)
            self.result_text.setText(result)

            pixmap = QPixmap(image_path)
            self.image_label.setPixmap(pixmap)

    def classify_variety(self, image_path):
        
        model = utils.get_resnet(None)

        param_dict = mindspore.load_checkpoint(f"./model/model_epoch_200.ckpt")
        mindspore.load_param_into_net(model, param_dict)

        preprocessed_image = utils.preprocess_single_image(image_path)


        # Convert the preprocessed image to a Tensor
        input_data = Tensor(preprocessed_image, dtype=mindspore.float32)

        # Run the model
        prediction = model(input_data)

        # Convert the prediction to a numpy array
        prediction = prediction.asnumpy()

        # Get the predicted class index
        predicted_class = np.argmax(prediction)



        # Get the predicted breed name
        predicted_breed = breed_mapping[predicted_class]
        #predicted_breed = predicted_class
        
        return predicted_breed

if __name__ == "__main__":


    data_folder = '../data/train_valid_test/train'

    breed_mapping = {}
    class_idx = 0
    file = os.listdir(data_folder)
    file = sorted(file)
    print(file)
    for folder_name in file:
        if os.path.isdir(os.path.join(data_folder, folder_name)):
            breed_mapping[class_idx] = folder_name
            class_idx += 1

    
    app = QApplication(sys.argv)
    window = VarietyClassifierApp()
    window.show()
    sys.exit(app.exec_())
