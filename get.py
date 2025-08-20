import requests
from PIL import Image
from io import BytesIO
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import os
from layers import L1Dist

app = Flask(__name__)

# Suppress TensorFlow warnings (optional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load TensorFlow/Keras model
model = tf.keras.models.load_model('siamesemodel.h5', custom_objects={'L1Dist': L1Dist})
# model.compile(optimizer='adam', loss='binary_crossentropy')

def preprocess_image(image):
    """Preprocesses the image."""
    img = image.resize((100, 100))
    img = np.array(img) / 255.0  # Normalize image
    return img

def get_image_from_url(image_url):
    """Fetches an image from the given URL."""
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # Raise error if request was not successful
        image = Image.open(BytesIO(response.content))
        return image
    except Exception as e:
        raise RuntimeError(f"Error fetching image from URL: {str(e)}")

def get_data_from_api(group_id):
    """Gets data from the API using the provided group ID."""
    url = 'http://localhost:3000/loginimage'
    params = {'groupId': group_id}

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise error if request was not successful
        data = response.json()
        image_url = data['images'][0]['imageBase64']
        return image_url
    except Exception as e:
        raise RuntimeError(f"Error fetching data from API: {str(e)}")

def get_data_from_api_all(group_id):
    """Gets data from the API using the provided group ID."""
    url = 'http://localhost:3000/getter'
    params = {'groupId': group_id}

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise error if request was not successful
        data = response.json()
        image_url = data['images']
        print(image_url)
        return image_url
    except Exception as e:
        raise RuntimeError(f"Error fetching data from API: {str(e)}")
    

@app.route('/get_data', methods=['GET'])
def get_data_verify():
    # Extracting query parameters
    group_id = request.args.get('groupId')
    group_login_id=request.args.get('groupLoginId')
    try:
        # Get data from API
        image_url = get_data_from_api(group_id)



        images_all = get_data_from_api_all(group_login_id)

        
        # Fetch image from URL
        image = get_image_from_url(image_url)

        # Preprocess the image
        img = preprocess_image(image)

        # Build results array
        results = []
        for filename in os.listdir(os.path.join('application_data', 'verification_images')):
            validation_img = preprocess_image(Image.open(os.path.join('application_data', 'verification_images', filename)))
            result = model.predict(list(np.expand_dims([img, validation_img], axis=1)))
            results.append(result)

        detection_threshold = 0.4
        verification_threshold = 0.6
        detection = np.sum(np.array(results) > detection_threshold)
        verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images')))
        verified = verification > verification_threshold

        response = {
            'detection': int(detection),  # Convert to int
            'verification': float(verification),  # Convert to float
            'verified': bool(verified) ,
            
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()
