from flask import Flask, render_template, request
from PIL import Image
import numpy as np
from mindspore import context, Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import utils
import mindspore as ms
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision


app = Flask(__name__)

net = utils.get_net(None)
param_dict = load_checkpoint('model/model_epoch_100(2).ckpt')
load_param_into_net(net, param_dict)
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

import os

data_folder = r"C:\Users\Lenovo\Desktop\huawei\Untitled Folder\d2l-zh\pytorch\chapter_computer-vision\data\kaggle_dog_tiny\train_valid_test\train"  # 修改为你的数据文件夹的路径

breed_mapping = {}
class_idx = 0

for folder_name in os.listdir(data_folder):
    if os.path.isdir(os.path.join(data_folder, folder_name)):
        breed_mapping[class_idx] = folder_name
        class_idx += 1



def preprocess_single_image(image_path):
    # Load the image
    image = Image.open(image_path)

    # Define preprocessing transformations
    transform_test = transforms.Compose([
    vision.Resize(256),
    # 从图像中心裁切224x224大小的图片
    vision.CenterCrop(224),
    vision.Normalize([0.485 * 255, 0.456 * 255, 0.406 * 255],
                     [0.229 * 255, 0.224 * 255, 0.225 * 255]),
    vision.HWC2CHW()])

    # Apply the transformations
    processed_image = transform_test(image)
    return processed_image


def predict_dog_breed(image_path):
    # Preprocess the image
    preprocessed_image = preprocess_single_image(image_path)

    # Convert the preprocessed image to a Tensor
    input_data = Tensor(preprocessed_image, dtype=ms.float32)

    # Run the model
    prediction = net(input_data)

    # Convert the prediction to a numpy array
    prediction = prediction.asnumpy()

    # Get the predicted class index
    predicted_class = np.argmax(prediction)



    # Get the predicted breed name
    predicted_breed = breed_mapping[predicted_class]

    return predicted_breed

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        image = request.files["image"]
        if image:
            image_path = os.path.join("static/temp_uploads", image.filename)
            image.save(image_path)
            predicted_breed = predict_dog_breed(image_path)
            return render_template("index.html", prediction=predicted_breed, image_url=image.filename)
    return render_template("index.html", prediction=None, image_url=None)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" in request.files:
        image = request.files["image"]
        if image:
            image_path = os.path.join("static/temp_uploads", image.filename)
            image.save(image_path)
            predicted_breed = predict_dog_breed(image_path)
            return render_template("index.html", prediction=predicted_breed, image_url=image.filename)
    return "Prediction failed"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
