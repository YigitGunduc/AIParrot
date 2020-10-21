import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from data_preprocessing import preprocessing
import numpy as np
import data_preprocessing
import model
import embedding

BATCH_SIZE = 64
EPOCHS = 100
LSTM_NODES =256
NUM_SENTENCES = 20000
MAX_SENTENCE_LENGTH = 25
MAX_NUM_WORDS = 11000
EMBEDDING_SIZE = 100

input_texts, target_texts, target_texts_inputs = data_preprocessing.preprocessing()

embedding_matrix, word2idx_inputs, word2idx_outputs, num_words, max_out_len, max_input_len, num_words_output = embedding.embedder(input_texts, target_texts, target_texts_inputs)

model = model.seq2seq_model(num_words, max_out_len, max_input_len, num_words_output)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

for j in range(0, len(input_texts), 100):

    train_text = input_texts[j:j+100]
    train_target = target_texts[j:j+100]
    train_text_inputs = target_texts_inputs[j:j+100]

    encoder_input_sequences, decoder_input_sequences, decoder_targets_one_hot = embedding.embedder_data(train_text, train_target, train_text_inputs,embedding_matrix, word2idx_inputs, word2idx_outputs, num_words, max_out_len, max_input_len, num_words_output)
    print(encoder_input_sequences.shape,decoder_input_sequences.shape, decoder_targets_one_hot.shape)
    r = model.fit([encoder_input_sequences, decoder_input_sequences],decoder_targets_one_hot,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.1,
        callbacks=[callback]
    )
    model.save(f"seq2seq{j}.h5")

print("training has done")
