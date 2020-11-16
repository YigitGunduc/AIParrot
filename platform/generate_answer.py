import re
import string
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
import utils

latent_dim = 256  # Latent dimensionality of the encoding space.
max_encoder_seq_length =  107 # longest sentence encoder will encounter
max_decoder_seq_length =  112 # longers sentence decoder will encounter

num_tokens, token_index, _ = utils.character_set()

def tokenizer(input_texts, max_encoder_seq_length, num_tokens):
    encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_tokens), dtype='float32')
    for i, input_text in enumerate(input_texts):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, token_index[char]] = 1.
    return encoder_input_data  



input_characters = sorted(list(re.sub(r'[A-Z]', '', string.printable)))
input_characters = sorted(list(re.sub(r'[A-Z]', '', string.printable)))
num_tokens = len(input_characters)


token_index = dict([(char, i) for i, char in enumerate(input_characters)])

# Restore the model and construct the encoder and decoder.
model = load_model('C:\\Users\\gunduc\\Desktop\\parrot-repos\\parrot\\platform\\seq2seq10000.h5')

encoder_inputs = model.input[0]   # input_1
encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output   # lstm_1
encoder_states = [state_h_enc, state_c_enc]
encoder_model = Model(encoder_inputs, encoder_states)

decoder_inputs = model.input[1]   # input_2
decoder_state_input_h = Input(shape=(latent_dim,), name='input_3')
decoder_state_input_c = Input(shape=(latent_dim,), name='input_4')
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = model.layers[3]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = model.layers[4]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)
# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_char_index = dict((i, char) for char, i in token_index.items())


# Decodes an input sequence.  Future work should support beam search.
def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


quest = input("type your question : ")
input_seq = tokenizer([quest], max_encoder_seq_length, num_tokens)
decoded_sentence = decode_sequence(input_seq)
print('Decoded sentence:', decoded_sentence)
