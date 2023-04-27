import os
import io
import requests
from firebase_admin import credentials, initialize_app, storage
import firebase_admin as admin
from flask import Flask, request, jsonify, send_file
# from keras.models import load_model
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from datetime import datetime, timedelta
from flask_cors import CORS, cross_origin
import urllib
import urllib.request

urllib.request.urlretrieve('https://media.githubusercontent.com/media/tarush-r/Background-BGone-ML/main/model.h5', 'model.h5')
model = tf.keras.models.load_model('./model.h5')


cred = credentials.Certificate(
    './keystone-295p-firebase-adminsdk-elgsn-2497393882.json')
initialize_app(cred, {'storageBucket': 'keystone-295p.appspot.com'})
bucket = storage.bucket()


app = Flask(__name__)
CORS(app)


@app.route('/segment', methods=['GET'])
def segment_image():

    img_id = request.args.get('img_id')
    user_id = request.args.get('user_id')
    print('printing')
    print(user_id)
    print(img_id)
    # blob = bucket.blob(f'https://storage.googleapis.com/keystone-295p.appspot.com/users/{user_id}/original/{img_id}.png')
    blob = bucket.blob(f'users/{user_id}/original/{img_id}')
    expiration_time = datetime.utcnow() + timedelta(minutes=5)
    img_url = blob.generate_signed_url(expiration=expiration_time)
    # img_url = blob.public_url
    # img_url = "https://firebasestorage.googleapis.com/v0/b/keystone-295p.appspot.com/o/users%2F1%2Foriginal%2F1.png?alt=media&token=bed10e87-36f1-4762-aee4-829ab99b0925"

    img_data = download_image_from_firebase(img_url)
    img_arr = np.array(Image.open(io.BytesIO(img_data)))

    image = cv2.cvtColor(img_arr, cv2.COLOR_BGRA2BGR)

    image = preprocess_image(image)

    seg_mask = model.predict(image)

    output = postprocess_prediction(seg_mask, img_arr)

    #output = Image.fromarray(np.uint8(seg_img))

    new_img_url = upload_image_to_firebase(output, user_id, img_id)

    response = jsonify({'new_image_url': new_img_url})
    response.headers.add("Access-Control-Allow-Origin", "*")

    return response


def download_image_from_firebase(img_url):

    img_data = requests.get(img_url).content
    return img_data


def preprocess_image(image):

    image = cv2.resize(image, (256, 256))
    image = image/255.0
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)
    return image


def postprocess_prediction(seg_mask, img_arr):

    seg_mask = seg_mask[0, :, :, -1]
    seg_mask = cv2.resize(seg_mask, (img_arr.shape[1], img_arr.shape[0]))
    seg_mask = np.expand_dims(seg_mask, axis=-1)
    seg_img = (img_arr * seg_mask).astype(np.uint8)
    seg_img = Image.fromarray(np.uint8(seg_img))
    seg_img = seg_img.convert('RGBA')
    pixel_data = seg_img.load()
    for i in range(seg_img.size[0]):   
        for j in range(seg_img.size[1]):   
            if pixel_data[i,j] == (0, 0, 0, 255):  # If black, make transparent
                pixel_data[i,j] = (0, 0, 0, 0)

    return seg_img


def upload_image_to_firebase(img, user_id, img_id):

    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')

    img_bytes.seek(0)

    new_filename = img_id + '.png'

    blob = bucket.blob(f'users/{user_id}/segmented/' + new_filename)
    blob.upload_from_file(img_bytes, content_type='image/png')

    expiration_time = datetime.utcnow() + timedelta(minutes=5)
    new_img_url = blob.generate_signed_url(expiration=expiration_time)

    #new_img_url = f"https://storage.googleapis.com/keystone-295p.appspot.com/users/{user_id}/segmented/{img_id}.png"

    return new_img_url


if __name__ == '__main__':
    app.run(debug=True)


# import os
# import io
# import requests
# from firebase_admin import credentials, initialize_app, storage
# import firebase_admin as admin
# from flask import Flask, request, jsonify, send_file
# # from keras.models import load_model
# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import cv2
# from datetime import datetime, timedelta
# from flask_cors import CORS, cross_origin




# model = tf.keras.models.load_model('./model.h5')


# cred = credentials.Certificate('../keystone-295p-firebase-adminsdk-elgsn-2497393882.json')
# initialize_app(cred, {'storageBucket': 'keystone-295p.appspot.com'})
# bucket = storage.bucket()


# app = Flask(__name__)
# CORS(app)

# @app.route('/segment', methods=['GET'])
# def segment_image():

#     img_id = request.args.get('img_id')
#     user_id = request.args.get('user_id')
#     # blob = bucket.blob(f'https://storage.googleapis.com/keystone-295p.appspot.com/users/{user_id}/original/{img_id}.png')
#     blob = bucket.blob(f'users/{user_id}/original/{img_id}')
#     expiration_time = datetime.utcnow() + timedelta(minutes=5)
#     img_url = blob.generate_signed_url(expiration=expiration_time)
#     # img_url = blob.public_url
#     # img_url = "https://firebasestorage.googleapis.com/v0/b/keystone-295p.appspot.com/o/users%2F1%2Foriginal%2F1.png?alt=media&token=bed10e87-36f1-4762-aee4-829ab99b0925"

#     img_data = download_image_from_firebase(img_url)
#     img_arr = np.array(Image.open(io.BytesIO(img_data)))

#     image = cv2.cvtColor(img_arr, cv2.COLOR_BGRA2BGR)

#     image = preprocess_image(image)

#     seg_mask = model.predict(image)

#     seg_img = postprocess_prediction(seg_mask, img_arr)

#     output = Image.fromarray(np.uint8(seg_img))

#     new_img_url = upload_image_to_firebase(output, user_id, img_id)

#     response = jsonify({'new_image_url': new_img_url})
#     response.headers.add("Access-Control-Allow-Origin", "*")

#     return response
# # @app.route('/segment', methods=['GET'])
# # def segment_image():
# #     print(' 1 ')
# #     img_id = request.json['image_id']
# #     user_id = request.json['user_id']
# #     # blob = bucket.blob(f'https://storage.googleapis.com/keystone-295p.appspot.com/users/{user_id}/original/{img_id}.png')
# #     blob = bucket.blob(f'users/{user_id}/original/{img_id}.png')
# #     expiration_time = datetime.utcnow() + timedelta(minutes=5)
# #     img_url = blob.generate_signed_url(expiration=expiration_time)
# #     # img_url = blob.public_url
# #     print(img_url)
# #     # img_url = "https://firebasestorage.googleapis.com/v0/b/keystone-295p.appspot.com/o/users%2F1%2Foriginal%2F1.png?alt=media&token=bed10e87-36f1-4762-aee4-829ab99b0925"


# #     img_data = download_image_from_firebase(img_url)
# #     img_arr = np.array(Image.open(io.BytesIO(img_data)))
# #     print(' 2 ')
# #     image = cv2.cvtColor(img_arr, cv2.COLOR_BGRA2BGR)
    
# #     image = preprocess_image(img_arr)
# #     print(' 3 ')
# #     seg_mask = model.predict(image)
    
# #     seg_img = postprocess_prediction(seg_mask, img_arr)
# #     print(' 4 ')
# #     output = Image.fromarray(np.uint8(seg_img))
    
# #     new_img_url = upload_image_to_firebase(output, user_id, img_id)
# #     print('returned response')
# #     response = jsonify({'new_image_url': new_img_url})
# #     response.headers.add("Access-Control-Allow-Origin", "*")
# #     return response

# def download_image_from_firebase(img_url):
  
#     img_data = requests.get(img_url).content
#     return img_data

# def preprocess_image(image):

#     image = cv2.resize(image, (256, 256))
#     image = image/255.0
#     image = image.astype(np.float32)
#     image = np.expand_dims(image, axis=0)
#     return image

# def postprocess_prediction(seg_mask, img_arr):
   
#     seg_mask = seg_mask[0,:,:,-1]
#     seg_mask = cv2.resize(seg_mask, (img_arr.shape[1], img_arr.shape[0]))
#     seg_mask = np.expand_dims(seg_mask, axis=-1)
#     seg_img = (img_arr * seg_mask).astype(np.uint8)
#     return seg_img

# def upload_image_to_firebase(img, user_id, img_id):
    
#     img_bytes = io.BytesIO()
#     img.save(img_bytes, format='PNG')
    
#     img_bytes.seek(0)
    
#     new_filename = img_id + '.png'
    
#     blob = bucket.blob(f'users/{user_id}/segmented/' + new_filename)
#     blob.upload_from_file(img_bytes, content_type='image/png')
    
#     new_img_url = f"https://storage.googleapis.com/keystone-295p.appspot.com/users/{user_id}/segmented/{img_id}.png"
    
#     return new_img_url

# if __name__ == '__main__':
#     app.run(debug=True)