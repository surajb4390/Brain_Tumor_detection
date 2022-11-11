from flask import Flask,render_template,request
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import cv2





app = Flask(__name__)

cnn = load_model('model.hdf5')

@app.route('/')
def input():
    return render_template('image_upload.html')


@app.route('/prediction', methods=["POST"])
def prediction():
    img = request.files['img']
    img.save('img.jpg')
    
    image = cv2.imread('img.jpg')
    test_img = cv2.resize(image,(64,64))/255
    test_input=test_img.reshape((1,64,64,3))
    

    pred = cnn.predict(test_input)[0]
    
    max_prob = max(pred)
   
    for i,prob in enumerate(pred):
       if prob==max_prob:
           if i==0:
               pred='Brain Tumor Belongs to Gilioma'
               
           elif i==1:
               pred='Brain Tumor Belongs to Meningiloma'
               
           else:
               pred='Brain Tumor Belongs to Piturity'


    return render_template('output.html',data=pred)    

if __name__=="__main__":
    app.run()