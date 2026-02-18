# Simple RNN for IMDB Sentiment Analysis

This project demonstrates the implementation of a Simple Recurrent Neural Network (RNN) to perform sentiment analysis on the IMDB movie review dataset. It includes data preprocessing, model training, and a Streamlit-based web application for real-time predictions.

## Project Structure

* **`simplernn.ipynb`**: The primary notebook used for training the Simple RNN model. It covers loading the IMDB dataset, word indexing, sequence padding, and model compilation using a ReLU activation and Sigmoid output.
* **`embedding.ipynb`**: A notebook exploring word embedding representations, one-hot encoding, and feature representation with TensorFlow/Keras.
* **`prediction.ipynb`**: A notebook designed to load the saved model and perform sentiment classification on sample user reviews.
* **`main.py`**: A Streamlit application script that provides a user interface for entering movie reviews and receiving a sentiment prediction (Positive or Negative) based on the trained model.
* **`simple_rnn_imdb1.h5`**: The pre-trained Keras model file.

## Model Details

The model architecture consists of:

1. **Embedding Layer**: Maps vocabulary indices to a dense vector space (128 dimensions).
2. **SimpleRNN Layer**: 128 units with ReLU activation.
3. **Dense Layer**: 1 unit with Sigmoid activation for binary classification.

The model was trained using the Adam optimizer and binary cross-entropy loss, incorporating Early Stopping to prevent overfitting.

## How to Run

### Prerequisites

Ensure you have the following libraries installed:

* NumPy
* Pandas
* TensorFlow
* Streamlit

### Running the Web App

To launch the sentiment analysis interface, run the following command in your terminal:

```bash
streamlit run main.py

```

### Usage

1. Enter a movie review in the text area provided by the Streamlit app.
2. Click the **Classify** button.
3. The app will display the predicted sentiment (Positive/Negative) and the corresponding prediction score.
