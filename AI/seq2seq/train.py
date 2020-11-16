"""
train.py depends on :
    -model.py
    -data_preprocessing
    -utils
    -numpy
    -tensorflow
train.py calls the data_preprocessing.py for conversations,
utils.py for num_tokens(num of charectes)
model.py for seq2seq model
"""
import tensorflow as tf
import numpy as np
import data_preprocessing
import model
import utils


batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 100000

num_tokens, token_index, _ = utils.character_set()
input_texts, target_texts, max_encoder_seq_length, max_decoder_seq_length = data_preprocessing.preprocessing()

max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

model = model.seq2seq_model(num_tokens)

model.compile(optimizer="rmsprop", loss='categorical_crossentropy', metrics=['accuracy'])

for j in range(0, len(input_texts), 10000):

    train_text = input_texts[j:j+10000]
    train_target = target_texts[j:j+10000]

    encoder_input_data = np.zeros(
        (len(train_text), max_encoder_seq_length, len(num_tokens)),
        dtype='float32')
    decoder_input_data = np.zeros(
        (len(train_text), max_decoder_seq_length, len(num_tokens)),
        dtype='float32')
    decoder_target_data = np.zeros(
        (len(train_text), max_decoder_seq_length, len(num_tokens)),
        dtype='float32')

    for i, (input_text, target_text) in enumerate(zip(train_text, train_target)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, token_index[char]] = 1.
            encoder_input_data[i, t + 1:, token_index[' ']] = 1.
        for t, char in enumerate(target_text):
            # decoder_target_data is ahead of decoder_input_data by one timestep
            decoder_input_data[i, t, token_index[char]] = 1.
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                decoder_target_data[i, t - 1, token_index[char]] = 1.
        decoder_input_data[i, t + 1:, token_index[' ']] = 1.
        decoder_target_data[i, t:, token_index[' ']] = 1.

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)

    model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=batch_size, epochs=200, callbacks=[callback], validation_split=0.1)
    model.save(f"seq2seq{j}.h5")
    print(f"Batch number {j} done")
print("training has done")
