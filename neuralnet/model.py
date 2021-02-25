import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Bidirectional, Embedding
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Input
from tensorflow.keras.models import load_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Lambda
from config import VOCAB_SIZE, MAXLEN 

def seq2seq():
    encoder_inputs= Input(shape=(25,))
    encoder_embedding = Embedding(VOCAB_SIZE, 100, input_length=MAXLEN)

    decoder_embedding = Embedding(VOCAB_SIZE, 100, input_length=MAXLEN)
    encoder_embeddings = encoder_embedding(encoder_inputs)
    encoder_lstm=LSTM(256, return_state=True, kernel_regularizer=l2(0.0000001), activity_regularizer=l2(0.0000001))
    LSTM_outputs, state_h, state_c = encoder_lstm(encoder_embeddings)

    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(25,), name='decoder_inputs')
    decoder_lstm = LSTM(256, return_sequences=True, return_state=True, name='decoder_lstm', kernel_regularizer=l2(0.0000001), activity_regularizer=l2(0.0000001))
    decoder_embeddings = decoder_embedding(decoder_inputs)
    decoder_outputs, _, _ = decoder_lstm(decoder_embeddings,
                                         initial_state=encoder_states)


    decoder_dense = Dense(5000, activation='softmax', name='decoder_dense')
    decoder_outputs = decoder_dense(decoder_outputs)

    seq2seq = Model([encoder_inputs, decoder_inputs], decoder_outputs, name='model_encoder_training')

    return seq2seq

