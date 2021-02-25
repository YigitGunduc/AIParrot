from math import log
import numpy as np
import tensorflow as tf
from config import LEARNING_RATE
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import LSTM, Embedding
from preprocessing import clean_text
from tokenizers import Tokenizer 
from config import VOCAB_SIZE, MAXLEN

# +++++++++++++++++++++++++++++++++ seq2seq model to refere layers with their names ++++++++++++++++++++++++++++++++
encoder_inputs = Input(shape=(25,))
encoder_embedding = Embedding(VOCAB_SIZE, 100, input_length=MAXLEN)

decoder_embedding = Embedding(VOCAB_SIZE, 100, input_length=MAXLEN)
encoder_embeddings = encoder_embedding(encoder_inputs)
encoder_lstm = LSTM(256, return_state=True, kernel_regularizer=l2(0.0000001), activity_regularizer=l2(0.0000001))
LSTM_outputs, state_h, state_c = encoder_lstm(encoder_embeddings)

encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(25,), name='decoder_inputs')
decoder_lstm = LSTM(256, return_sequences=True, return_state=True, name='decoder_lstm', kernel_regularizer=l2(0.0000001), activity_regularizer=l2(0.0000001))
decoder_embeddings = decoder_embedding(decoder_inputs)
decoder_outputs, _, _ = decoder_lstm(decoder_embeddings, initial_state=encoder_states)

decoder_dense = Dense(5000, activation='softmax', name='decoder_dense')
decoder_outputs = decoder_dense(decoder_outputs)

Seq2SeqModel = Model([encoder_inputs, decoder_inputs], decoder_outputs, name='model_encoder_training')


# +++++++++++++++++++++++++++++++++ model for predictions +++++++++++++++++++++++++++++++++ 
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(256,))
decoder_state_input_c = Input(shape=(256,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]


decoder_inputs = Input(shape=(1,))
embedded = decoder_embedding(decoder_inputs)
decoder_outputs, state_h, state_c = decoder_lstm(embedded, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
                    [decoder_inputs] + decoder_states_inputs,
                    [decoder_outputs] + decoder_states)


# +++++++++++++++++++++++++++++++++ Predict Class +++++++++++++++++++++++++++++++++ 
class Predict():
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def create_response(self, question):
        question = np.expand_dims(self.tokenizer.tokenize_sequence(clean_text(question)), axis=0)
        result = self.predict_sentence(question)
        return result 

 
    def predict_sentence(self, input_seq):
        with tf.device('/cpu:0'):
            states_value = encoder_model.predict(input_seq)

            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = self.tokenizer.tokenizer.word_index['<sos>']
            output_sentence = []

            for _ in range(MAXLEN):
                output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
                idx = np.argmax(output_tokens)

                if self.tokenizer.tokenizer.index_word[idx] == '<eos>':

                    break

                output_sentence.append(idx)
                target_seq[0, 0] = idx
                states_value = [h, c]

        return self.tokenizer.decode_sequence(output_sentence)

