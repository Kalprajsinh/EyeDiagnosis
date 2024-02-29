# from flask import Flask, request, jsonify
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np

# app = Flask(__name__)

# # Load the trained model
# model = load_model(r'C:\Users\91878\OneDrive\Desktop\eyemove1\model.h5')  # Update with your model file path

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Receive image from Flutter app
#     image_file = request.files['image']
    
#     # Read and preprocess the image
#     img = image.load_img(image_file, target_size=(224, 224))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array /= 255.0  # Normalize pixel values
    
#     # Make prediction
#     prediction = model.predict(img_array)
    
#     # Process prediction result
#     if prediction[0][0] >= 0.5:
#         result = "Normal Eye"
#     else:
#         result = "Cataract detected"
    
#     # Return prediction result
#     return jsonify({'result': result})

# if __name__ == '__main__':
#     app.run(debug=True)

########################################################################## for HTML
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = load_model(r'model.h5')  # Update with your model file path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the POST request has the file part
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')

        file = request.files['file']

        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return render_template('index.html', error='No selected file')

        if file:
            # Save the uploaded file to a temporary location
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Read and preprocess the image
            img = image.load_img(file_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Normalize pixel values

            # Make prediction
            prediction = model.predict(img_array)

            # Process prediction result
            if prediction[0][0] >= 0.5:
                result = "Normal Eye"
            else:
                result = "Cataract detected"

            return render_template('index.html', result=result)

    return render_template('index.html')

if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = 'uploads'  # Create a folder named 'uploads' in your project directory
    app.run(host="0.0.0.0",port=5000)

# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.




