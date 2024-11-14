# app.py
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import io
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('best_model.h5')

# Image preprocessing function
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize((128, 128))
    img = np.array(img) / 255.0  # Normalize
    return np.expand_dims(img, axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    img_bytes = file.read()
    img = preprocess_image(img_bytes)
    
    # Make prediction
    prediction = model.predict(img)
    class_label = 'cataract' if prediction[0][0] > 0.5 else 'normal'
    confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]
    
    return jsonify({'prediction': class_label, 'confidence': confidence})

if __name__ == '__main__':
    app.run(debug=True)
