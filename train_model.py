# ================================
# Mental Health Sentiment Analysis
# Simple RNN Training Script
# ================================

# Install dataset package if needed
# pip install kagglehub

import os
import pickle
import numpy as np
import pandas as pd
import kagglehub

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import matplotlib.pyplot as plt
import seaborn as sns

# =========================================
# STEP 1 — LOAD DATASET
# =========================================

print("Downloading dataset...")

path = kagglehub.dataset_download(
    "suchintikasarkar/sentiment-analysis-for-mental-health"
)

print("Dataset path:", path)

files = os.listdir(path)

print("Files in dataset folder:")
print(files)

# Load CSV
df = pd.read_csv(os.path.join(path, files[0]))

print("\nDataset Loaded Successfully!")

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nSentiment Classes:")
print(df['status'].unique())

print("\nClass Distribution:")
print(df['status'].value_counts())

# =========================================
# STEP 2 — TEXT PREPROCESSING
# =========================================

VOCAB_SIZE = 10000
MAX_LENGTH = 100
PADDING_TYPE = 'post'
TRUNC_TYPE = 'post'

# Handle missing values
df['statement'] = df['statement'].astype(str).fillna('')

# Create tokenizer
tokenizer = Tokenizer(
    num_words=VOCAB_SIZE,
    oov_token="<OOV>"
)

# Fit tokenizer
tokenizer.fit_on_texts(df['statement'])

# Convert text to sequences
sequences = tokenizer.texts_to_sequences(df['statement'])

# Pad sequences
padded_sequences = pad_sequences(
    sequences,
    maxlen=MAX_LENGTH,
    padding=PADDING_TYPE,
    truncating=TRUNC_TYPE
)

print("\nText preprocessing completed!")

print("Padded Sequence Shape:", padded_sequences.shape)

# =========================================
# STEP 3 — LABEL ENCODING
# =========================================

label_encoder = LabelEncoder()

encoded_labels = label_encoder.fit_transform(df['status'])

print("\nEncoded Classes:")
print(label_encoder.classes_)

# =========================================
# STEP 4 — TRAIN TEST SPLIT
# =========================================

X_train, X_val, y_train, y_val = train_test_split(
    padded_sequences,
    encoded_labels,
    test_size=0.2,
    random_state=42
)

print("\nTraining Shape:", X_train.shape)
print("Validation Shape:", X_val.shape)

# =========================================
# STEP 5 — BUILD SIMPLE RNN MODEL
# =========================================

num_classes = df['status'].nunique()

EMBEDDING_DIM = 128

model = Sequential([
    
    Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=EMBEDDING_DIM
    ),

    SimpleRNN(
        units=128,
        return_sequences=True
    ),

    SimpleRNN(
        units=128
    ),

    Dense(
        units=num_classes,
        activation='softmax'
    )
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Show model summary
print("\nModel Summary:")
model.summary()

# =========================================
# STEP 6 — TRAIN MODEL
# =========================================

EPOCHS = 10
BATCH_SIZE = 64

print("\nStarting Training...\n")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

print("\nTraining Completed!")

# =========================================
# STEP 7 — EVALUATION
# =========================================

print("\nEvaluating Model...")

# Predictions
y_pred_probs = model.predict(X_val)

y_pred = np.argmax(y_pred_probs, axis=1)

# Accuracy
accuracy = accuracy_score(y_val, y_pred)

print(f"\nAccuracy: {accuracy:.4f}")

# Precision Recall F1
precision, recall, f1_score, _ = precision_recall_fscore_support(
    y_val,
    y_pred,
    average='weighted'
)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")

# =========================================
# STEP 8 — CONFUSION MATRIX
# =========================================

conf_matrix = confusion_matrix(y_val, y_pred)

plt.figure(figsize=(10, 8))

sns.heatmap(
    conf_matrix,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()

# =========================================
# STEP 9 — REAL TIME PREDICTION FUNCTION
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

    # Prediction
    prediction_probs = model.predict(padded_sequence)

    predicted_class = np.argmax(prediction_probs, axis=1)[0]

    predicted_label = label_encoder.inverse_transform(
        [predicted_class]
    )[0]

    confidence = np.max(prediction_probs)

    print("\n================================")
    print("Input Text:", text)
    print("Predicted Sentiment:", predicted_label)
    print(f"Confidence Score: {confidence:.4f}")
    print("================================")


# Example Predictions
predict_sentiment(
    "I feel very depressed and lonely."
)

predict_sentiment(
    "Today is an amazing day and I feel happy."
)

predict_sentiment(
    "I am stressed about my exams."
)

# =========================================
# STEP 10 — SAVE MODEL + FILES
# =========================================

print("\nSaving Model Files...")

# Save model
model.save("rnn_sentiment_model.keras")

# Save tokenizer
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Save label encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("\nAll files saved successfully!")

print("\nGenerated Files:")
print("1. rnn_sentiment_model.keras")
print("2. tokenizer.pkl")
print("3. label_encoder.pkl")

print("\nProject Completed Successfully!")
