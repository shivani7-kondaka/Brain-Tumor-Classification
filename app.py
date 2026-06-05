from flask import Flask, request, render_template, send_from_directory
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import json
import os

app = Flask(__name__)

model = load_model('brain_tumor_model.h5')
with open('class_names.json','r') as f:
    CLASS_NAMES = json.load(f)
image_size = 128

UPLOAD = "./upload"
if not os.path.exists(UPLOAD):
    os.makedirs(UPLOAD)
app.config['UPLOAD'] = UPLOAD

def dandd(image_path):
    image_size =128
    img = load_img(image_path, target_size=(image_size, image_size))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    pred_idx = np.argmax(predictions)
    confidence = np.max(predictions) * 100

    label = CLASS_NAMES[pred_idx]
    result =  "No Tumor" if label == 'notumor' else f"Tumor: {label}"
    return result, confidence
   

@app.route("/", methods = ['GET','POST'])
def index():
   
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_loc = os.path.join(app.config['UPLOAD'],file.filename)
            file.save(file_loc)

            result, confidence = dandd(file_loc)
            return render_template('index.html', result = result, confidence=f'{confidence:.2f}', file_path = f'/upload/{file.filename}')
    return render_template('index.html', result=None)

@app.route('/upload/<filename>')
def get_upfile(filename):
    return send_from_directory(app.config['UPLOAD'],filename)

if __name__ == '__main__':
    app.run(debug=True)
