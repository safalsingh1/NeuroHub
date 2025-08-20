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

def get_data_from_api(group_id,url):
    """Gets data from the API using the provided group ID."""
    url = url+'/loginimage'
    params = {'groupId': group_id}

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise error if request was not successful
        data = response.json()
        image_url = data['images'][0]['imageBase64']
        return image_url
    except Exception as e:
        raise RuntimeError(f"Error fetching data from API: {str(e)}")

def get_data_from_api_all(group_id,url):
    """Gets data from the API using the provided group ID."""
    url = url+'/getter'
    params = {'groupId': group_id}

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise error if request was not successful
        data = response.json()
        image_urls = [img['imageBase64'] for img in data['images']]
        return image_urls
    except Exception as e:
        raise RuntimeError(f"Error fetching data from API: {str(e)}")

@app.route('/get_data', methods=['GET'])
def get_data_verify():
    # Extracting query parameters
    group_id = request.args.get('groupId')
    group_login_id = request.args.get('groupLoginId')
    url=request.args.get('url')
    try:
        # Get data from API
        image_url = get_data_from_api(group_id,url)
        images_all = get_data_from_api_all(group_login_id,url)
        
        # Fetch image from URL
        image = get_image_from_url(image_url)
        # Preprocess the image
        img = preprocess_image(image)

        # Build results array
        results = []
        for img_url in images_all:
            validation_image = get_image_from_url(img_url)
            validation_img = preprocess_image(validation_image)
            result = model.predict(list(np.expand_dims([img, validation_img], axis=1)))
            results.append(result)
        
        detection_threshold = 0.3
        verification_threshold = 0.4
        detection = np.sum(np.array(results) > detection_threshold)
        verification = detection / len(images_all)
        verified = verification > verification_threshold
        print(result)
        response = {
            'detection': int(detection),  # Convert to int
            'verification': float(verification),  # Convert to float
            'verified': bool(verified),
            'imageurks': images_all,
           
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# New route that sends "Hello" when called
@app.route('/hello', methods=['GET'])
def say_hello():
    return "Hello", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0')
