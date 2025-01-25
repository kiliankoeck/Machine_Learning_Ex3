import random
import pickle
import heapq

import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop
from pathlib import Path

import argparse


def parse_command():
    parser = argparse.ArgumentParser(description="Process some data.")

    parser.add_argument("-i", "--input", type=str, help="Path to the input file.")
    #TODO: add output file
    parser.add_argument("-o", "--output", type=str, help="Path to the output file.")
    parser.add_argument("-n", "--n_best", type=int, default=1, help="Amount of suggestions to give.")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Amount of epochs to train model.")

    args = parser.parse_args()

    return args.input, args.output, args.n_best, args.epochs


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
        if word in unique_token_index:  # Handle unknown words
            X[0, i, unique_token_index[word]] = 1
        else:
            print(f"Warning: Word '{word}' not found in vocabulary. Ignoring.")

    predictions = model.predict(X)[0]
    indices = np.argpartition(predictions, -n_best)[-n_best:]
    predicted_words = [unique_tokens[idx] for idx in indices]
    return predicted_words


def load_model_with_metadata(input_file, epochs):
    model_path = Path("nw_model.keras")
    metadata_path = Path("nw_metadata.pkl")

    if model_path.is_file() and metadata_path.is_file():
        while True:
            print("An existing model was found.")
            user_input = input("Do you want to use it? [Y/n]: ").strip().lower()

            if user_input == 'y':
                model = tf.keras.models.load_model(model_path)
                with open(metadata_path, "rb") as f:
                    metadata = pickle.load(f)
                return model, metadata
            elif user_input == 'n':
                print("Training a new model will override the existing model.")
                user_input = input("Do you want to continue? [Y/n]: ").strip().lower()
                if user_input == 'y':
                    model = train_model(input_file, epochs)
                    with open("nw_metadata.pkl", "rb") as f:
                        metadata = pickle.load(f)
                    return model, metadata
                else:
                    return None, None
    else:
        model = train_model(input_file, epochs)
        with open("nw_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        return model, metadata


def train_model(input_file, epochs):

    if(input_file is None):
        print("No input file was provided. Exiting.")
        exit()

    print("Training a new model.")

    tokens, n_words = extract_tokens(input_file)
    unique_tokens = np.unique(tokens)
    X, y, unique_tokens, unique_token_index, n_words = prep_data(tokens, unique_tokens, n_words)
    model = build_model(unique_tokens, n_words)
    model.fit(X, y, batch_size=128, epochs=epochs)
    model.save('nw_model.keras')

    metadata = {
        "unique_tokens": unique_tokens,
        "unique_token_index": unique_token_index,
        "n_words": n_words,
    }
    with open("nw_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    return model


def extract_tokens(input_file):
    text_df = pd.read_csv(input_file)
    text = list(text_df.text.values)
    joined_text = " ".join(text)

    with open("joined_text.txt", "w", encoding="utf-8") as f:
        f.write(joined_text)

    partial_text = joined_text[:1000000]

    tokenizer = RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(partial_text.lower())

    n_words = 10

    return tokens, n_words


def prep_data(tokens, unique_tokens, n_words):
    input_words = []
    next_word = []

    unique_token_index = {token: index for index, token in enumerate(unique_tokens)}

    for i in range(len(tokens) - n_words):
        input_words.append(tokens[i:i + n_words])
        next_word.append(tokens[i + n_words])

    X = np.zeros((len(input_words), n_words, len(unique_tokens)),
                 dtype=bool)  # for each sample, n input words and then a boolean for each possible next word
    y = np.zeros((len(next_word), len(unique_tokens)),
                 dtype=bool)  # for each sample a boolean for each possible next word

    for i, words in enumerate(input_words):
        for j, word in enumerate(words):
            X[i, j, unique_token_index[word]] = 1
        y[i, unique_token_index[next_word[i]]] = 1

    return X, y, unique_tokens, unique_token_index, n_words


def build_model(unique_tokens, n_words):
    model = Sequential()
    model.add(LSTM(128, input_shape=(n_words, len(unique_tokens))))
    model.add(Dense(len(unique_tokens)))
    model.add(Activation("softmax"))

    optimizer = RMSprop(learning_rate=0.01)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    return model


if __name__ == "__main__":
    input_file, output_file, n_best, epochs = parse_command()
    model, metadata = load_model_with_metadata(input_file, epochs)
    if model and metadata:
        predict(model, metadata, n_best)
    else:
        print("No model was loaded or trained. Exiting.")
