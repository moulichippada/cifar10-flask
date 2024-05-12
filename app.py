from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
model = load_model('cifar10_model1.keras')

# Define class names for CIFAR-10
class_names = ['aeroplane', 'Automobile', 'Bird', 'Cat', 'Deer','Dog', 'Frog', 'Horse', 'Ship', 'Truck']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(request.url)

    img_file = request.files['image']

    if img_file.filename == '':
        return redirect(request.url)
    print('path',img_file)
    img_file.save(img_file.filename)

    try:
        img = image.load_img(img_file.filename, target_size=(32, 32))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = class_names[predicted_class_index]
        print(prediction)

        return render_template('index.html', prediction=predicted_class_name)
    except Exception as e:
        error_message = f"Error processing image: {e}"
        print(error_message)
        return render_template('index.html', error=error_message)

if __name__ == '__main__':
    app.run(port=5000,debug=True)
    
