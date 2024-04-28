from django.views.generic.edit import CreateView
from django.views.generic.base import TemplateView
from django.shortcuts import render
from .models import CoverLetterGenerator
from .forms import CoverLetterGeneratorForm
from django.urls import reverse_lazy
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
import os


# Download NLTK resources (if not downloaded already)
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Determine the absolute path to the "pcl.csv" file
file_path = os.path.join(os.path.dirname(__file__), "pcl.csv")

df = pd.read_csv(file_path)


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


class HomeView(TemplateView):
    template_name = "home.html"


class CoverLetterView(CreateView):
    form_class = CoverLetterGeneratorForm
    model = CoverLetterGenerator
    template_name = "cover-letter.html"
    success_url = reverse_lazy("cover")

    def form_valid(self, form):
        link = form.cleaned_data["project_description"]
        print(link)
        # Load the trained model
        # Determine the absolute path to the "pcl.csv" file
        file_path_2 = os.path.join(
            os.path.dirname(__file__), "cover_letter_generator_model.h5"
        )
        model = tf.keras.models.load_model(file_path_2)

        # New project description
        new_project_description = [link]

        # Preprocess the new project description
        preprocessed_new_project_description = preprocess_text(
            new_project_description[0]
        )

        # Convert the preprocessed project description to sequences using the tokenizer
        new_project_sequence = tokenizer_desc.texts_to_sequences(
            [preprocessed_new_project_description]
        )

        # Pad the sequence to match the input sequence length used during training
        new_project_padded_sequence = pad_sequences(
            new_project_sequence, maxlen=max_seq_length, padding="post"
        )

        # Predict the cover letter for the new project description
        predicted_cover_letter_sequence = model.predict(new_project_padded_sequence)

        # Convert the predicted sequence back to text using the tokenizer for cover letters
        predicted_cover_letter_text = []
        for sequence in predicted_cover_letter_sequence[
            0
        ]:  # Take the first sequence (as there's only 1)
            # Sample a token based on its probability distribution
            sampled_token_index = np.random.choice(len(sequence), p=sequence)
            # Convert the index to its corresponding word
            word = tokenizer_cover.index_word.get(sampled_token_index, "")
            # Append the word to the cover letter text
            predicted_cover_letter_text.append(word)

        # Join the words to form the predicted cover letter text
        predicted_cover_letter_text = " ".join(predicted_cover_letter_text)

        # Print the generated cover letter
        # print("Generated Cover Letter:")
        # print(predicted_cover_letter_text)
        predicted_cover_letter_text = " ".join(
            predicted_cover_letter_text.split()
        )  # Remove extra whitespace
        words = predicted_cover_letter_text.split()
        half_length = len(words) // 2
        first_paragraph = " ".join(words[:half_length])
        second_paragraph = " ".join(words[half_length:])

        print(first_paragraph)

        self.object = form.save()
        return render(
            self.request,
            self.template_name,
            {
                "form": form,
                "first_paragraph": first_paragraph,
                "second_paragraph": second_paragraph,
            },
        )
