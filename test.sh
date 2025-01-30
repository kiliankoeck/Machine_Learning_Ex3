#!/bin/bash

VENV_PATH="./venv/Scripts/activate"
INPUT_FILE=".\fake_and_real_news_dataset.csv"

EPOCHS=(10 20 30)
BATCH_SIZES=(128 256)
LSTM_UNITS=(64 128)
TEXT_LENGTHS=(10000 100000)
LEARNING_RATES=(0.01 0.001)

source "$VENV_PATH"

for epoch in "${EPOCHS[@]}"; do
  for batch_size in "${BATCH_SIZES[@]}"; do
    for lstm_units in "${LSTM_UNITS[@]}"; do
      for text_length in "${TEXT_LENGTHS[@]}"; do
        for learn_rate in "${LEARNING_RATES[@]}"; do
          echo "Training with EPOCHS=$epoch, BATCH_SIZE=$batch_size, LSTM_UNITS=$lstm_units, TEXT_LENGTH=$text_length, LEARNING_RATE=$learn_rate"
          
          python train_model.py -i "$INPUT_FILE" -e "$epoch" -b "$batch_size" -lu "$lstm_units" -t "$text_length" -lr "$learn_rate" -nw 10
          
          echo "Finished training with EPOCHS=$epoch, BATCH_SIZE=$batch_size, LSTM_UNITS=$lstm_units, TEXT_LENGTH=$text_length, LEARNING_RATE=$learn_rate"
          echo "---------------------------------------------"
        done
      done
    done
  done
done
