from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

MODEL_PATH = "models\\traffic_sign_model.h5"
model = None

## Class names for 43 traffic signs (GTSRB dataset)
class_names = [
    'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)',
    'Speed limit (60km/h)', 'Speed limit (70km/h)', 'Speed limit (80km/h)',
    'End of speed limit (80km/h)', 'Speed limit (100km/h)', 'Speed limit (120km/h)',
    'No passing', 'No passing for vehicles over 3.5 metric tons',
    'Right-of-way at the next intersection', 'Priority road', 'Yield',
    'Stop', 'No vehicles', 'Vehicles over 3.5 metric tons prohibited',
    'No entry', 'General caution', 'Dangerous curve to the left',
    'Dangerous curve to the right', 'Double curve', 'Bumpy road',
    'Slippery road', 'Road narrows on the right', 'Road work',
    'Traffic signals', 'Pedestrians', 'Children crossing',
    'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing',
    'End of all speed and passing limits', 'Turn right ahead',
    'Turn left ahead', 'Ahead only', 'Go straight or right',
    'Go straight or left', 'Keep right', 'Keep left',
    'Roundabout mandatory', 'End of no passing',
    'End of no passing by vehicles over 3.5 metric tons'
]

def load_model():
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
            print("Model loaded successfully!")
        
        else:
            print("Model file not found")

    except Exception as e:
        print(f"Error loading the model: {e}")


def preprocess_img(image):
    img = image.resize((32, 32))
    img_array = np.array(img)
    img_array = img_array.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# Call load_model to load the model when app starts
load_model()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]

    ## Read and process image
    image_bytes = file.read()
    image = Image.open(io.BytesIO(image_bytes))

    ## Converting the image to RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    ## Preprocessing the image
    processed_image = preprocess_img(image)

    ## Making predictions
    predictions = model.predict(processed_image, verbose=0)

    ## Getting top prediction
    predicted_class = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class])

    ## Getting top 3 predictions
    top_3_indices = np.argsort(predictions[0])[-3:][::-1]
    top_3_predictions = [
            {
                "class": class_names[i],
                "class_id": int(i),
                "confidence": float(predictions[0][i]),
                "confidence_percentage": f"{float(predictions[0][i]) * 100:.2f}%"
            }
            for i in top_3_indices
        ]
    
    return render_template("predict.html", 
                       prediction_class=class_names[predicted_class],
                       confidence=f"{confidence * 100:.2f}%",
                       top_3_predictions=top_3_predictions)