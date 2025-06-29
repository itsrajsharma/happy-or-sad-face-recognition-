from flask import Flask, request, render_template
import numpy as np
import os
import uuid

app = Flask(__name__)

# Global variable for model (will be loaded when needed)
model = None

def load_model_safely():
    """Load the model with error handling"""
    global model
    if model is None:
        try:
            from tensorflow.keras.models import load_model as keras_load_model
            model = keras_load_model('hpy_sad_01.h5')
            print("Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    return True

def preprocess_image(file_path):
    try:
        from tensorflow.keras.preprocessing import image
        img = image.load_img(file_path, target_size=(256, 256))
        img = image.img_to_array(img)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        raise Exception("Error processing the uploaded image.")

def predict_emotion(img_array):
    """Make prediction with error handling"""
    try:
        if not load_model_safely():
            return "Model not available"
        
        prediction = model.predict(img_array)
        prediction_value = prediction[0][0]
        print(f"Prediction Value: {prediction_value}")
        result = "Sad" if prediction[0][0] > 0.5 else "Happy"
        return result
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        return "Prediction failed"

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling image uploads
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error="No file selected")
    
    file = request.files['file']
    if file.filename == '' or file.filename is None:
        return render_template('index.html', error="No file selected")
    
    # Check if file is an image
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    if not file.filename.lower().endswith(tuple('.' + ext for ext in allowed_extensions)):
        return render_template('index.html', error="Please upload an image file (PNG, JPG, JPEG, GIF, BMP)")
    
    try:
        # Generate unique filename to prevent overwrites
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4().hex}{file_extension}"
        file_path = os.path.join('static', unique_filename)
        
        # Save the file
        file.save(file_path)

        # Preprocess the image
        img_array = preprocess_image(file_path)

        # Predict the emotion
        result = predict_emotion(img_array)
        return render_template('result.html', result=result, image=unique_filename)
    
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return render_template('index.html', error="Error processing image. Please try again.")

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)

