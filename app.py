import streamlit as st

# =========================================
# PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND)
# =========================================

st.set_page_config(
    page_title="MindPulse AI",
    page_icon="🧠",
    layout="centered"
)

# =========================================
# IMPORTS
# =========================================

import tensorflow as tf
import pickle
import numpy as np
import os
import gdown

from tensorflow.keras.preprocessing.sequence import pad_sequences

# =========================================
# DOWNLOAD MODEL FROM GOOGLE DRIVE
# =========================================

MODEL_URL = "https://drive.google.com/uc?id=1x_mnmJj4KrwKzoWPkrFpbjYZciRxCqQI"

MODEL_FILE = "rnn_sentiment_model.keras"

# Download model if not exists
if not os.path.exists(MODEL_FILE):

    with st.spinner("Downloading AI model... Please wait."):

        gdown.download(
            MODEL_URL,
            MODEL_FILE,
            quiet=False
        )

# =========================================
# LOAD MODEL
# =========================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_FILE)

model = load_model()

# =========================================
# LOAD TOKENIZER
# =========================================

@st.cache_resource
def load_tokenizer():

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    return tokenizer

tokenizer = load_tokenizer()

# =========================================
# LOAD LABEL ENCODER
# =========================================

@st.cache_resource
def load_label_encoder():

    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    return label_encoder

label_encoder = load_label_encoder()

# =========================================
# CONSTANTS
# =========================================

MAX_LENGTH = 100
PADDING_TYPE = 'post'
TRUNC_TYPE = 'post'

# =========================================
# PREDICTION FUNCTION
# =========================================

def predict_sentiment(text):

    # Convert text to sequence
    sequence = tokenizer.texts_to_sequences([text])

    # Pad sequence
    padded_sequence = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding=PADDING_TYPE,
        truncating=TRUNC_TYPE
    )

    # Predict
    prediction = model.predict(padded_sequence)

    predicted_class = np.argmax(prediction, axis=1)[0]

    predicted_label = label_encoder.inverse_transform(
        [predicted_class]
    )[0]

    confidence = np.max(prediction)

    return predicted_label, confidence

# =========================================
# UI
# =========================================

st.title("🧠 MindPulse AI")

st.markdown(
    """
    ### Mental Health Sentiment Analysis
    
    This AI model predicts mental health sentiment
    from user text using a Simple RNN model.
    """
)

# Text area
user_input = st.text_area(
    "Enter Your Text",
    height=150,
    placeholder="Type how you feel..."
)

# Predict button
if st.button("Predict Sentiment"):

    if user_input.strip() == "":

        st.warning("Please enter some text.")

    else:

        with st.spinner("Analyzing sentiment..."):

            label, confidence = predict_sentiment(user_input)

        # Show result
        st.success(f"Predicted Sentiment: {label}")

        st.info(f"Confidence Score: {confidence:.2f}")

# =========================================
# FOOTER
# =========================================

st.markdown("---")

st.caption(
    "Built with TensorFlow, Streamlit, and Simple RNN"
)
