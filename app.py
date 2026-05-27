import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model
model = tf.keras.models.load_model("rnn_sentiment_model.keras")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Constants
MAX_LENGTH = 100
PADDING_TYPE = 'post'
TRUNC_TYPE = 'post'

# Prediction function
def predict_sentiment(text):
    sequence = tokenizer.texts_to_sequences([text])

    padded_sequence = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding=PADDING_TYPE,
        truncating=TRUNC_TYPE
    )

    prediction = model.predict(padded_sequence)

    predicted_class = np.argmax(prediction, axis=1)[0]

    predicted_label = label_encoder.inverse_transform([predicted_class])[0]

    confidence = np.max(prediction)

    return predicted_label, confidence

# Streamlit UI
st.title("🧠 Mental Health Sentiment Analysis")

st.write("Enter a sentence to predict mental health sentiment.")

user_input = st.text_area("Enter Text")

if st.button("Predict"):

    if user_input.strip() != "":

        label, confidence = predict_sentiment(user_input)

        st.success(f"Predicted Sentiment: {label}")

        st.info(f"Confidence Score: {confidence:.2f}")

    else:
        st.warning("Please enter some text.")
