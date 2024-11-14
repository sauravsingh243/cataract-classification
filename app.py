import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
from PIL import Image
import io


app = Flask(__name__)
model = tf.keras.models.load_model('best_model.h5')

def prepare_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        image = Image.open(io.BytesIO(file.read()))
        processed_image = prepare_image(image)
        prediction = model.predict(processed_image)[0][0]
        class_label = 'Cataract' if prediction > 0.5 else 'No Cataract'
        confidence = float(prediction) if class_label == 'Cataract' else 1 - float(prediction)

        return jsonify({
            'class': class_label,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
