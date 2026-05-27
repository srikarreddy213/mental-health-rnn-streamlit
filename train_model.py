# =========================================
# MINDPULSE AI — FINAL SIMPLE RNN MODEL
# =========================================

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

from tensorflow.keras.layers import (
    Embedding,
    SimpleRNN,
    Dense,
    Dropout,
    Bidirectional
)

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import (
    pad_sequences
)

from tensorflow.keras.callbacks import (
    EarlyStopping
)

from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import seaborn as sns

# =========================================
# DOWNLOAD DATASET
# =========================================

print("Downloading dataset...")

path = kagglehub.dataset_download(
    "suchintikasarkar/sentiment-analysis-for-mental-health"
)

print("Dataset Path:", path)

files = os.listdir(path)

print("Files:", files)

# =========================================
# LOAD DATASET
# =========================================

df = pd.read_csv(
    os.path.join(path, files[0])
)

print("\nDataset Loaded!")

print("\nDataset Shape:")
print(df.shape)

print("\nClasses:")
print(df['status'].unique())

# =========================================
# CLEAN DATA
# =========================================

# Remove missing values
df['statement'] = (
    df['statement']
    .astype(str)
    .fillna('')
)

# Remove short sentences
df = df[
    df['statement']
    .str
    .split()
    .str
    .len() > 3
]

# Shuffle dataset
df = df.sample(
    frac=1,
    random_state=42
)

print("\nCleaned Dataset Shape:")
print(df.shape)

# =========================================
# PREPROCESSING
# =========================================

VOCAB_SIZE = 15000

MAX_LENGTH = 120

PADDING_TYPE = 'post'

TRUNC_TYPE = 'post'

# =========================================
# TOKENIZER
# =========================================

tokenizer = Tokenizer(
    num_words=VOCAB_SIZE,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(
    df['statement']
)

# Convert text into sequences
sequences = tokenizer.texts_to_sequences(
    df['statement']
)

# Padding
padded_sequences = pad_sequences(
    sequences,
    maxlen=MAX_LENGTH,
    padding=PADDING_TYPE,
    truncating=TRUNC_TYPE
)

print("\nPadded Shape:")
print(padded_sequences.shape)

# =========================================
# LABEL ENCODING
# =========================================

label_encoder = LabelEncoder()

encoded_labels = label_encoder.fit_transform(
    df['status']
)

print("\nEncoded Classes:")
print(label_encoder.classes_)

# =========================================
# TRAIN TEST SPLIT
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
# BUILD OPTIMIZED SIMPLE RNN MODEL
# =========================================

num_classes = df['status'].nunique()

EMBEDDING_DIM = 128

model = Sequential([

    Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=EMBEDDING_DIM,
        input_length=MAX_LENGTH
    ),

    Bidirectional(

        SimpleRNN(
            128,
            return_sequences=True
        )

    ),

    Dropout(0.3),

    Bidirectional(

        SimpleRNN(
            64
        )

    ),

    Dropout(0.3),

    Dense(
        64,
        activation='relu'
    ),

    Dense(
        units=num_classes,
        activation='softmax'
    )
])

# =========================================
# COMPILE MODEL
# =========================================

model.compile(
    optimizer=Adam(
        learning_rate=0.001
    ),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nMODEL SUMMARY\n")

model.summary()

# =========================================
# EARLY STOPPING
# =========================================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# =========================================
# TRAIN MODEL
# =========================================

EPOCHS = 12

BATCH_SIZE = 64

print("\nStarting Training...\n")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop]
)

print("\nTraining Completed!")

# =========================================
# EVALUATION
# =========================================

print("\nEvaluating Model...")

y_pred_probs = model.predict(X_val)

y_pred = np.argmax(
    y_pred_probs,
    axis=1
)

# Accuracy
accuracy = accuracy_score(
    y_val,
    y_pred
)

print(f"\nAccuracy: {accuracy:.4f}")

# Precision Recall F1
precision, recall, f1_score, _ = (
    precision_recall_fscore_support(
        y_val,
        y_pred,
        average='weighted'
    )
)

print(f"Precision: {precision:.4f}")

print(f"Recall: {recall:.4f}")

print(f"F1 Score: {f1_score:.4f}")

# =========================================
# CONFUSION MATRIX
# =========================================

conf_matrix = confusion_matrix(
    y_val,
    y_pred
)

plt.figure(figsize=(12, 10))

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
# REAL TIME PREDICTION FUNCTION
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

    print("\n================================")

    print("Input:", text)

    print("Prediction:", predicted_label)

    print(f"Confidence: {confidence:.4f}")

    print("================================")

# =========================================
# TEST PREDICTIONS
# =========================================

predict_sentiment(
    "I feel very sad and lonely today"
)

predict_sentiment(
    "I am extremely anxious about exams"
)

predict_sentiment(
    "I feel happy and excited today"
)

predict_sentiment(
    "I feel depressed and hopeless"
)

# =========================================
# SAVE MODEL + FILES
# =========================================

print("\nSaving files...")

model.save(
    "rnn_sentiment_model.keras"
)

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("\nFiles Saved Successfully!")

print("\nGenerated Files:")

print("1. rnn_sentiment_model.keras")

print("2. tokenizer.pkl")

print("3. label_encoder.pkl")

print("\nPROJECT COMPLETED SUCCESSFULLY!")
