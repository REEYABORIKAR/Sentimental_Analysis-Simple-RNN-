import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Load word index
with open("word_index.pkl", "rb") as f:
    word_index = pickle.load(f)

reverse_word_index = {value: key for key, value in word_index.items()}

# Load model
model = load_model("simple_rnn_imdb1.h5")

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

st.title("IMDB Movie Review Sentiment Analysis")
user_input = st.text_area("Movie Review")

if st.button("Classify"):
    if user_input.strip():
        preprocessed_input = preprocess_text(user_input)
        prediction = model.predict(preprocessed_input)
        sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"

        st.success(f"Sentiment: {sentiment}")
        st.write(f"Prediction score: {prediction[0][0]:.4f}")
    else:
        st.warning("Please enter a review.")
