
import requests
from PIL import Image
from io import BytesIO
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import os
from layers import L1Dist

# Suppress TensorFlow warnings (optional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)
app.config['USE_RELOADER'] = False
# Load TensorFlow/Keras model
model = tf.keras.models.load_model('siamesemodel.h5', custom_objects={'L1Dist': L1Dist})
# model.compile(optimizer='adam', loss='binary_crossentropy')

def preprocess(file_path):
    """Load image from file and convert to 100x100px."""
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (100, 100))
    img = img / 255.0
    return img

@app.route('/verify', methods=['POST'])
def verify():
    if not request.is_json:
        return jsonify({"error": "Request content type must be application/json"}), 400
    
    # Specify thresholds
    detection_threshold = 0.4
    verification_threshold = 0.6

    data = request.get_json()
    image_url = data.get('image_url')
    
    # Fetch image from URL
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # Raise error if request was not successful
        image = Image.open(BytesIO(response.content))
    except Exception as e:
        return jsonify({"error": f"Error fetching image from URL: {str(e)}"}), 400

    # Resize and preprocess the image
    img = image.resize((100, 100))
    img = np.array(img) / 255.0  # Normalize image
    
    # Build results array
    results = []
    for filename in os.listdir(os.path.join('application_data', 'verification_images')):
        validation_img = preprocess(os.path.join('application_data', 'verification_images', filename))
        
        result = model.predict(list(np.expand_dims([img, validation_img], axis=1)))
        results.append(result)
    
    detection = np.sum(np.array(results) > detection_threshold)
    verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images')))
    verified = verification > verification_threshold

    response = {
        'detection': int(detection),  # Convert to int
        'verification': float(verification),  # Convert to float
        'verified': bool(verified)  # Convert to bool
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run()
