import pickle
import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from keras import Input
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop
from pathlib import Path

import argparse

def parse_command():
    parser = argparse.ArgumentParser(description="Load a trained model and generate text predictions.")

    parser.add_argument("-m", "--model_path", required=True, type=str, help="Path to model (.keras file).")
    parser.add_argument("-n", "--n_best", type=int, default=10, help="Amount of suggestions to give.")

    args = parser.parse_args()
    return args.model_path, args.n_best


def predict(model, metadata, n_best):
    unique_tokens = metadata["unique_tokens"]
    unique_token_index = metadata["unique_token_index"]
    n_words = metadata["n_words"]

    while True:
        user_input = input("Enter your sentence or type exit to quit: ").strip()
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        else:
            predictions = predict_next_word(
                model, user_input, n_best, unique_tokens, unique_token_index, n_words
            )
            print("Predicted next words:", predictions)


def predict_next_word(model, input_text, n_best, unique_tokens, unique_token_index, n_words):
    input_text = input_text.lower()
    X = np.zeros((1, n_words, len(unique_tokens)))
    for i, word in enumerate(input_text.split()):
        if word in unique_token_index:
            X[0, i, unique_token_index[word]] = 1
        else:
            print(f"Warning: Word '{word}' not found in vocabulary. Ignoring.")

    predictions = model.predict(X)[0]
    indices = np.argpartition(predictions, -n_best)[-n_best:]
    predicted_words = [str(unique_tokens[idx]) for idx in indices]
    return predicted_words


def load_model_with_metadata(model_file, metadata_file):
    if model_file.is_file() and metadata_file.is_file():
        model = tf.keras.models.load_model(model_file)
        with open(metadata_file, "rb") as f:
            metadata = pickle.load(f)
        return model, metadata
    else:
        raise FileNotFoundError(f"Model file {model_file} or metadata file {metadata_file} does not exist.")


if __name__ == "__main__":
    model_path, n_best = parse_command()

    model_file = Path(f"{model_path}")
    metadata_file = Path(f"{model_path.replace(".keras", ".pkl").replace("model", "metadata")}")

    try:
        model, metadata = load_model_with_metadata(model_file, metadata_file)
        print("Model and metadata loaded successfully.")
        print(f"The model uses {metadata["n_words"]} words for prediction.")

        predict(model, metadata, n_best)
    except FileNotFoundError as e:
        print(e)
