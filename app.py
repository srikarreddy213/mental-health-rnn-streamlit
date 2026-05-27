import streamlit as st
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

model = tf.keras.models.load_model(MODEL_FILE)

# =========================================
# LOAD TOKENIZER
# =========================================

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# =========================================
# LOAD LABEL ENCODER
# =========================================

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

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

    # Convert text into sequence
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
# STREAMLIT PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="Mental Health Sentiment Analysis",
    page_icon="🧠",
    layout="centered"
)

# =========================================
# UI DESIGN
# =========================================

st.title("🧠 Mental Health Sentiment Analysis")

st.markdown(
    """
    This AI model predicts mental health sentiment
    from user text using a Simple RNN model.
    """
)

# Text input
user_input = st.text_area(
    "Enter Your Text Here",
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

st.caption("Built with TensorFlow, Streamlit, and Simple RNN")
