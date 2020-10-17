from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from data_preprocessing import preprocessing
import embedding

BATCH_SIZE = 64
EPOCHS = 100
LSTM_NODES =256
NUM_SENTENCES = 20000
MAX_SENTENCE_LENGTH = 25
MAX_NUM_WORDS = 11000
EMBEDDING_SIZE = 100

input_texts, target_texts, target_texts_inputs = preprocessing()

embedding_matrix, word2idx_inputs, word2idx_outputs, num_words, max_out_len, max_input_len, num_words_output = embedding.embedder(input_texts, target_texts, target_texts_inputs)


embedding_layer = Embedding(num_words, EMBEDDING_SIZE, weights=[embedding_matrix], input_length=max_input_len,trainable= False)

def seq2seq_model(num_words, max_out_len, max_input_len, num_words_output):
    encoder_inputs_placeholder = Input(shape=(max_input_len,))
    x = embedding_layer(encoder_inputs_placeholder)
    encoder = LSTM(LSTM_NODES, return_state=True)
    encoder_outputs, h, c = encoder(x)
    encoder_states = [h, c]

    decoder_inputs_placeholder = Input(shape=(max_out_len,))
    decoder_embedding = Embedding(num_words_output, LSTM_NODES)
    decoder_inputs_x = decoder_embedding(decoder_inputs_placeholder)
    decoder_lstm = LSTM(LSTM_NODES, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs_x, initial_state=encoder_states)
    decoder_dense = Dense(num_words_output, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs_placeholder, decoder_inputs_placeholder], decoder_outputs)  
    return model
