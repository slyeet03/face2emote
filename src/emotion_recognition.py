import numpy as np
import tensorflow as tf
from PIL import Image
import cv2 as cv
from utils.config import IMG_HEIGHT, IMG_WIDTH, LABELS, MODEL_PATH

model = tf.keras.models.load_model(MODEL_PATH)

def process_image(image):
    try:
        if isinstance(image, np.ndarray):
            img = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2GRAY))
        else:
            img = Image.open(image)
            if img.mode != 'L':
                img = img.convert('L')
        
        # Resize to 48x48
        img = img.resize((IMG_HEIGHT, IMG_WIDTH))
        
        # Convert to numpy array
        img_array = np.array(img, dtype=np.float32)
        
        # Normalize pixel values to [0, 1]
        img_array = img_array / 255.0
        
        # Reshape to (1, IMG_HEIGHT, IMG_WIDTH, 1) for model input
        img_array = img_array.reshape(1, IMG_HEIGHT, IMG_WIDTH, 1)
        
        return img_array
        
    except Exception as e:
        print(f"Error processing image : {str(e)}")
        return None

def predict(img):
    img_array = process_image(img)
    prediction = model.predict(img_array, verbose=0)
    predicted_label = np.argmax(prediction[0])
    confidence = np.max(prediction[0])

    return predicted_label, confidence