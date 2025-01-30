import pickle
import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import os;

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from keras import Input
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop

import argparse


def parse_command():
    parser = argparse.ArgumentParser(description="Train a next word predicition model.")

    parser.add_argument("-i", "--input", required=True, type=str, help="Path to the input file.")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("-b", "--batch_size", type=int, default=128, help="Batch size for training.")
    parser.add_argument("-lu", "--lstm_units", type=int, default=128, help="Number of LSTM memory cells.")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.01, help="Learning rate for optimization.")
    parser.add_argument("-t", "--text_length", type=int, default=10000, help="Length of text to use from the dataset.")
    parser.add_argument("-nw", "--n_words", type=int, default=10, help="Number of words used as input for each training sample.")

    args = parser.parse_args()

    return args.input, args.epochs, args.batch_size, args.lstm_units, args.learning_rate, args.text_length, args.n_words


def extract_tokens(input_file, text_length):
    text_df = pd.read_csv(input_file)
    text = list(text_df.text.values)
    joined_text = " ".join(text)

    partial_text = joined_text[:text_length]

    tokenizer = RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(partial_text.lower())
    return tokens


def prep_data(tokens, unique_tokens, unique_token_index, n_words):
    input_words = []
    next_word = []

    for i in range(len(tokens) - n_words):
        input_words.append(tokens[i:i + n_words])
        next_word.append(tokens[i + n_words])

    X = np.zeros((len(input_words), n_words, len(unique_tokens)),
                 dtype=bool)
    y = np.zeros((len(next_word), len(unique_tokens)),
                 dtype=bool)

    for i, words in enumerate(input_words):
        for j, word in enumerate(words):
            X[i, j, unique_token_index[word]] = 1
        y[i, unique_token_index[next_word[i]]] = 1

    return X, y


if __name__ == "__main__":

    input_file, epochs, batch_size, n_lstm, learn_rate, text_length, n_words = parse_command()
    tokens = extract_tokens(input_file, text_length)
    unique_tokens = np.unique(tokens)
    unique_token_index = {token: index for index, token in enumerate(unique_tokens)}
    X, y = prep_data(tokens, unique_tokens, unique_token_index, n_words)

    model = Sequential([
        Input(shape=(n_words, len(unique_tokens))),
        LSTM(n_lstm),
        Dense(len(unique_tokens)),
        Activation("softmax"),
    ])

    optimizer = RMSprop(learning_rate=learn_rate)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    history = model.fit(X, y, batch_size=batch_size, epochs=epochs)
    last_epoch_accuracy = history.history['accuracy'][-1]

    formatted_lr = str(learn_rate).replace(".", "")
    model_name = f"results/model_e{epochs}_b{batch_size}_lstm{n_lstm}_lr{formatted_lr}_t{text_length}_nw{n_words}.keras"
    metadata_name = f"results/metadata_e{epochs}_b{batch_size}_lstm{n_lstm}_lr{formatted_lr}_t{text_length}_nw{n_words}.pkl"
    report_name = f"results/training_report_e{epochs}_b{batch_size}_lstm{n_lstm}_lr{formatted_lr}_t{text_length}_nw{n_words}.txt"

    model.save(model_name)

    metadata = {
        "unique_tokens": unique_tokens,
        "unique_token_index": unique_token_index,
        "n_words": n_words,
    }
    with open(metadata_name, "wb") as f:
        pickle.dump(metadata, f)

    with open(report_name, "w") as report:
        report.write(f"Training Report\n")
        report.write(f"----------------\n")
        report.write(f"Epochs: {epochs}\n")
        report.write(f"Batch Size: {batch_size}\n")
        report.write(f"LSTM Units: {n_lstm}\n")
        report.write(f"Learning Rate: {learn_rate}\n")
        report.write(f"Text Length used: {text_length}\n")
        report.write(f"Number of Words used: {n_words}\n")
        report.write(f"Final Accuracy: {last_epoch_accuracy:.4f}\n")

    print(f"Training complete. Model saved as {model_name}.")
    print(f"Metadata saved as {metadata_name}.")
    print(f"Report saved as {report_name}.")
