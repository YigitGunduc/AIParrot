from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense

latent_dim = 256  # Latent dimensionality of the encoding space.


def seq2seq_model(characters):
    encoder_inputs = Input(shape=(None, len(characters)))
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None, len(characters)))
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(len(characters), activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model
