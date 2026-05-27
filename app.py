import streamlit as st

# =========================================
# PAGE CONFIG
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

from tensorflow.keras.preprocessing.sequence import (
    pad_sequences
)

# =========================================
# GOOGLE DRIVE MODEL DOWNLOAD
# =========================================

MODEL_URL = "https://drive.google.com/uc?id=1D6TkEDy6dsgDS7aUdraAtFyfP09ZjHe6"

MODEL_FILE = "rnn_sentiment_model.keras"

# Download model if not present
if not os.path.exists(MODEL_FILE):

    with st.spinner("Downloading AI Model..."):

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

    model = tf.keras.models.load_model(
        MODEL_FILE
    )

    return model

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

MAX_LENGTH = 120

PADDING_TYPE = "post"

TRUNC_TYPE = "post"

# =========================================
# PREDICTION FUNCTION
# =========================================

def predict_sentiment(text):

    sequence = tokenizer.texts_to_sequences(
        [text]
    )

    padded_sequence = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding=PADDING_TYPE,
        truncating=TRUNC_TYPE
    )

    prediction_probs = model.predict(
        padded_sequence
    )

    predicted_class = np.argmax(
        prediction_probs,
        axis=1
    )[0]

    predicted_label = (
        label_encoder.inverse_transform(
            [predicted_class]
        )[0]
    )

    confidence = np.max(
        prediction_probs
    )

    return predicted_label, confidence

# =========================================
# CUSTOM CSS
# =========================================

st.markdown(
    """
    <style>

    .main {
        background-color: #0E1117;
    }

    .stTextArea textarea {
        font-size: 18px;
        border-radius: 12px;
    }

    .big-font {
        font-size: 22px !important;
        font-weight: bold;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =========================================
# UI
# =========================================

st.title("🧠 MindPulse AI")

st.markdown(
    """
    ## Mental Health Sentiment Analysis

    This AI model predicts mental health sentiment
    using an optimized Simple RNN Deep Learning model.
    """
)

# =========================================
# USER INPUT
# =========================================

user_input = st.text_area(
    "Enter Your Feelings or Thoughts",
    height=180,
    placeholder="Example: I feel anxious and stressed about my future..."
)

# =========================================
# BUTTON
# =========================================

if st.button("Predict Sentiment"):

    if user_input.strip() == "":

        st.warning(
            "Please enter some text."
        )

    else:

        with st.spinner(
            "Analyzing Mental State..."
        ):

            label, confidence = (
                predict_sentiment(
                    user_input
                )
            )

        # =========================================
        # RESULT
        # =========================================

        st.success(
            f"Predicted Sentiment: {label}"
        )

        st.info(
            f"Confidence Score: {confidence:.2f}"
        )

        # =========================================
        # CONFIDENCE BAR
        # =========================================

        st.progress(float(confidence))

# =========================================
# SIDEBAR
# =========================================

st.sidebar.title("🧠 About")

st.sidebar.info(
    """
    MindPulse AI is a Deep Learning based
    Mental Health Sentiment Analysis system
    built using:

    - TensorFlow
    - Simple RNN
    - Streamlit
    - NLP
    """
)

# =========================================
# SAMPLE INPUTS
# =========================================

st.markdown("---")

st.subheader("Example Inputs")

st.code(
    "I feel very depressed and lonely today"
)

st.code(
    "I am extremely anxious about my exams"
)

st.code(
    "I feel happy and excited today"
)

# =========================================
# FOOTER
# =========================================

st.markdown("---")

st.caption(
    "Built with ❤️ using TensorFlow and Streamlit"
)
