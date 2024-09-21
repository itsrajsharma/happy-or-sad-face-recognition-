from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)

# Load your model
model = load_model('hpy_sad_02.h5')

def preprocess_image(file_path):
    img = image.load_img(file_path, target_size=(256, 256))  # Resize to match model's input shape
    img = image.img_to_array(img)  # Convert to array
    img = img / 255.0  # Normalize the image to [0, 1] range
    img = np.expand_dims(img, axis=0)  # Add batch dimension (1, 256, 256, 3)
    return img


# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling image uploads
@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        # Save the file
        file_path = os.path.join('static', file.filename)
        file.save(file_path)

        # Preprocess the image
        img_array = preprocess_image(file_path)

        # Predict
        prediction = model.predict(img_array)
        prediction_value = prediction[0][0]
        print(f"Prediction Value: {prediction_value}")
        result = "Sad" if prediction[0][0] > 0.5 else "Happy"


        return render_template('result.html', result=result, image=file.filename)

if __name__ == "__main__":
    app.run(debug=True)

# model = load_model('imageclassifier.h5')
# model.summary()

