import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import re
import nltk

# Download NLTK resources (if not downloaded already)
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")


df = pd.read_csv("pcl.csv")


# Text preprocessing function
def preprocess_text(text):
    # Remove newline characters
    text = text.replace("**", "")
    text = text.replace("\n", " ")

    # Tokenization
    tokens = word_tokenize(text)
    # Remove punctuation
    table = str.maketrans("", "", string.punctuation)
    stripped = [word.translate(table) for word in tokens]
    # Remove non-alphabetic tokens
    words = [word for word in stripped if word.isalpha()]
    # Remove stop words
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if not word in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)


# Extract project descriptions and cover letters
project_descriptions = df["Project Description"].tolist()
cover_letters = df["Cover Letter"].tolist()

# Preprocess the project descriptions and cover letters
preprocessed_project_descriptions = [
    preprocess_text(desc) for desc in project_descriptions
]
preprocessed_cover_letters = [preprocess_text(letter) for letter in cover_letters]


# Tokenize input and output sequences
tokenizer_desc = Tokenizer()
tokenizer_desc.fit_on_texts(preprocessed_project_descriptions)
tokenizer_cover = Tokenizer()
tokenizer_cover.fit_on_texts(preprocessed_cover_letters)

# Convert text sequences to integer sequences
X = tokenizer_desc.texts_to_sequences(preprocessed_project_descriptions)
y = tokenizer_cover.texts_to_sequences(preprocessed_cover_letters)

# Pad sequences to ensure uniform length
max_seq_length = max(max(len(seq) for seq in X), max(len(seq) for seq in y))
X = pad_sequences(X, maxlen=max_seq_length, padding="post")
y = pad_sequences(y, maxlen=max_seq_length, padding="post")


# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Model architecture
def create_model(input_vocab_size, output_vocab_size, max_seq_length, hidden_units):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(input_vocab_size, hidden_units),
            tf.keras.layers.LSTM(hidden_units),
            tf.keras.layers.RepeatVector(max_seq_length),
            tf.keras.layers.LSTM(hidden_units, return_sequences=True),
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(output_vocab_size, activation="softmax")
            ),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    return model


# Create model
input_vocab_size = len(tokenizer_desc.word_index) + 1
output_vocab_size = len(tokenizer_cover.word_index) + 1
hidden_units = 256
model = create_model(input_vocab_size, output_vocab_size, max_seq_length, hidden_units)

# Train model
batch_size = 64
epochs = 3
model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    batch_size=batch_size,
    epochs=epochs,
)


# Save the trained model
model.save("cover_letter_generator_model.h5")
