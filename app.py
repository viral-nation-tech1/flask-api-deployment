
from flask import Flask, request, jsonify
from PIL import Image
import os
import json
import numpy as np
import os
import pickle
import joblib
import tensorflow as tf
from flask import Flask, request
from tensorflow.keras.models import load_model
# pip install Pillow
app = Flask(__name__)

@app.route("/im_size", methods=["POST"])
def process_image():
    model = load_model('effi_model1.h5')
    # model = joblib.load("effi_model1.h5")
    file = request.files['image']
    file.save('./test_image.jpg')
    # Read the image via file.stream
    imgage_file = Image.open(file.stream)
    print(imgage_file)
    # imgage_file.save(str(file))
    # imagefile = request.files.get(image_data_path, '')
    x_test=[]
    # directory = r'./test'
    # for filename in os.listdir(img):
    #     test_data= os.path.join(img, filename)
    #     x_test.append(test_data)
    #     print(x_test)

    y_hat = []
    # for i in x_test:
    # file = imgage_file
    imgage_file = './test_image.jpg'
    img_object = tf.keras.preprocessing.image.load_img(imgage_file,
                                                       target_size=(225, 225))
    img_array = tf.keras.preprocessing.image.img_to_array(img_object)
    img_array = tf.expand_dims(img_array, 0)
    # print("*"*50)
    # print("Image array is :")
    # print(img_array)
    # # print(model[5])
    # print(type(model))
    # print("*"*50)  
    predictions = model.predict(img_array)
    class_names = ['None', 'alcoholic_beverages', 'drinking',
                    'female_swimwear','nudity', 'revealing_clothes',
                    'smoking', 'weapons']
    y_pred=class_names[np.argmax(predictions[0])]
    # y_hat.append(y_pred)

    return jsonify({'msg': 'success','prediction':y_pred})


if __name__ == "__main__":
    app.run(debug=True)