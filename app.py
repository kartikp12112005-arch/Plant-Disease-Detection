import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("model/plant_disease_model.h5")

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

CLASS_NAMES = sorted(os.listdir("dataset"))

st.title(" Plant Disease Detection")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).resize((224,224))
    st.image(image, width="stretch")

    img = np.array(image)/255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    idx = np.argmax(pred)
    confidence = np.max(pred)*100

    st.success(f"Disease: {CLASS_NAMES[idx]}")
    st.info(f"Confidence: {confidence:.2f}%")
