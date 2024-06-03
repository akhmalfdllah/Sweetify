import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import io
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import pandas as pd
import json

from flask import Flask, request, jsonify

# Load model
model = keras.models.load_model("model.h5")

# Load grading data
df_grading = pd.read_csv('./Dataset/Data Minuman.csv')

# Load labels from JSON
with open('class_indices.json', 'r') as f:
    labels = json.load(f)

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).resize((150, 150))
    img = np.array(img).astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def get_nutrifacts(drink_name):
    drink = df_grading[df_grading['Produk'] == drink_name]
    drink = drink[['Gula/Sajian(g)', 'Gula/100ml(g)', 'Grade']].iloc[0]
    return drink

def predict(img_tensor):
    predictions = model.predict(img_tensor)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_label = labels[str(predicted_class_index)]
    return predicted_class_label

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({"error": "no file"})

        try:
            image_bytes = file.read()
            tensor = preprocess_image(image_bytes)
            prediction = predict(tensor)
            nutrifacts = get_nutrifacts(prediction)
            data = {
                "Gula/Sajian(g)": str(nutrifacts["Gula/Sajian(g)"]),
                "Gula/100ml(g)": str(nutrifacts["Gula/100ml(g)"]),
                "Grade": nutrifacts["Grade"],
                "Product": prediction
            }
            return jsonify(data)
        except Exception as e:
            return jsonify({"error": str(e)})

    return "OK we did it yoi"

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))