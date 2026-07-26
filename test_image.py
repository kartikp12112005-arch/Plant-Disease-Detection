import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array

# Load trained model
model = tf.keras.models.load_model("model/plant_disease_model.h5")

# Class names (ORDER SAME AS TRAINING)
class_names = [
    'Apple_Healthy',
    'Apple_Black_Rot',
    'Potato_Healthy',
    'Potato_Early_Blight',
    'Tomato_Healthy',
    'Tomato_Late_Blight'
]

# Load image (CORRECT PATH)
img = load_img("test_images/test.jpg", target_size=(224,224))
img = img_to_array(img) / 255.0
img = np.expand_dims(img, axis=0)

# Predict
prediction = model.predict(img)
predicted_class = class_names[np.argmax(prediction)]

print("✅ Predicted Disease:", predicted_class)
