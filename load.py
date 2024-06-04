import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from PIL import Image
import io
import json

# Set environment variable to suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Load model
model = keras.models.load_model('./model.h5')

# Load grading data
df_grading = pd.read_csv('./Dataset/Data Minuman.csv')

# Load labels from JSON
with open('./class_indices.json', 'r') as f:
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

# Load the image
with open('sample.JPG', 'rb') as file:
    image_bytes = file.read()

# Preprocess the image
img = preprocess_image(image_bytes)

# Perform inference
predictions = model.predict(img)
predicted_class_index = np.argmax(predictions, axis=1)[0]
predicted_class_label = labels[str(predicted_class_index)]

# Display the prediction
print(f'Predicted: {predicted_class_label}')

# Get nutritional facts
nutrifacts = get_nutrifacts(predicted_class_label)
print(f'Gula/100ml(g): {nutrifacts["Gula/100ml(g)"]}')
print(f'Grade: {nutrifacts["Grade"]}')