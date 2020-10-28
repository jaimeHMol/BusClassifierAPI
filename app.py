#!flask/bin/python
from flask import Flask, jsonify, request, abort

import cv2
import numpy as np
import base64
from pathlib import Path

import keras
from keras.models import load_model


app = Flask(__name__)
# OUTPUT_MODELS_PATH = Path('/home/jaimehmol/Projects/Python/BusClassifierWorkspace/BusClassifier/model/')
# MODEL = "busModel.h5"
OUTPUT_MODELS_PATH = Path('C:/Users/jaherran/Projects/Python/BusClassifier/vehicle/working/models/')
MODEL = "busModel.mod"


def readb64(base64_string):
    nparr = np.frombuffer(base64.b64decode(base64_string), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img




@app.route("/")
def index():
    return "Hi, this is the Bus Classifier home!"

@app.route("/BusClassifier/api/v1.0/predict", methods=["POST"])
def create_predict():
    if not request.json or not "input_img" in request.json:
        abort(400)

    class_names = ["Bicycle", "Boat", "Bus", "Car", "Motorcycle", "Truck", "Van"]
    encoded_string = request.json["input_img"]

    # Loading input image
    img = readb64(encoded_string)
    img = cv2.resize(img, (224,224))
    img_np = np.reshape(img, [1, 224, 224, 3])
    
    # Loading model
    model = load_model(OUTPUT_MODELS_PATH / MODEL)
    optimizer = keras.optimizers.Adam()
    model.compile(loss="categorical_crossentropy",
                    optimizer=optimizer,
                    metrics=["accuracy"])
    # Making prediction                    
    pred = model.predict(img_np)
    pred_formated = [ "{:.4f}".format(float(prediction*100)) for prediction in pred[0] ]

    
    # img_np_serialized = pickle.dumps(img_np, protocol = 0)
    # output = str(img_np_serialized)
    output = {class_name: value for class_name, value in zip(class_names, pred_formated)}
    return jsonify({"prediction_vector": output}), 201

if __name__ == "__main__":
    app.run(debug=True)