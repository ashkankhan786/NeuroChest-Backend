from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import io
import tensorflow as tf
from keras.utils import custom_object_scope
from tensorflow.keras.layers import Layer
from model_downloader import download_model
import os

app = Flask(__name__)
CORS(app)


class Cast(Layer):
    def __init__(self, dtype, **kwargs):
        super().__init__(**kwargs)
        self.target_dtype = tf.as_dtype(dtype)

    def call(self, inputs):
        return tf.cast(inputs, self.target_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"dtype": self.target_dtype.name})
        return config
    
custom_objects = {"Cast": Cast}

# Define the model URLs
model_urls = {
    'VGG16': 'https://huggingface.co/ashkankhan/chest-xray-models/resolve/main/model_VGG16.h5',
    'VGG19': 'https://huggingface.co/ashkankhan/chest-xray-models/resolve/main/model_VGG19.h5',
    'ResNet50': 'https://huggingface.co/ashkankhan/chest-xray-models/resolve/main/model_ResNet50.h5',
    'DenseNet121': 'https://huggingface.co/ashkankhan/chest-xray-models/resolve/main/model_DenseNet121.h5'
}

os.makedirs('./models', exist_ok=True)

# Load all 4 models (once at startup)
model_paths = []
for name, url in model_urls.items():
    local_path = f"./models/{name}.h5"
    download_model(url, local_path)
    model_paths.append(local_path)

models = []
with custom_object_scope(custom_objects):
    for path in model_paths:
        models.append(tf.keras.models.load_model(path))


# Define your disease labels in the same order as your output layer
LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia",
    "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other",
    "Fracture", "Support Devices"
]


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    img_bytes = file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    image = image.resize((224, 224))

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 224, 224, 3)

    # Predict with each model and collect results
    predictions = [model.predict(img_array)[0] for model in models]
    avg_prediction = np.mean(predictions, axis=0)

    # Get labels with probability > 0.5
    result = {
        label: float(prob)
        for label, prob in zip(LABELS, avg_prediction)
    }

    return jsonify({'prediction': result})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
