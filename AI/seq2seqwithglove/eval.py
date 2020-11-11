import numpy as np
import os
import re
import sys
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from numpy import array
from numpy import asarray
from numpy import zeros
import embedding
import data_preprocessing

BATCH_SIZE = 64
EPOCHS = 100
LSTM_NODES =256
NUM_SENTENCES = 20000
MAX_SENTENCE_LENGTH = 25
MAX_NUM_WORDS = 11000
EMBEDDING_SIZE = 100


def preprocessing():

    lines = open("C:\\Users\\gunduc\\Desktop\\parrot\\data/movie_lines.txt", encoding="utf-8", errors="ignore").read().split("\n")
    convers = open("C:\\Users\\gunduc\\Desktop\\parrot\\data\\movie_conversations.txt", encoding="utf-8", errors="ignore").read().split("\n")

    def clean_text(txt):
        txt = txt.lower()
        txt = re.sub(r"i'm", "i am", txt)
        txt = re.sub(r"it's", "it is", txt)
        txt = re.sub(r"he's", "he is", txt)
        txt = re.sub(r"she's", "she is", txt)
        txt = re.sub(r"that's", "that is", txt)
        txt = re.sub(r"what's", "what is", txt)
        txt = re.sub(r"where's", "where is", txt)
        txt = re.sub(r"\'ll", " will", txt)
        txt = re.sub(r"\'ve", " have", txt)
        txt = re.sub(r"\'re", " are", txt)
        txt = re.sub(r"\'d", " would", txt)
        txt = re.sub(r"won't", "will not", txt)
        txt = re.sub(r"can't", "can not", txt)

        return txt

    exchn = []
    for conver in convers:
        exchn.append(conver.split(" +++$+++ ")[-1][1:-1].replace("'", "").replace(",", "").split())

    diag = {}
    for line in lines:
        diag[line.split(" +++$+++ ")[0]] = line.split(" +++$+++ ")[-1]

    questions = []
    answers = []

    for conver in exchn:
        for i in range(len(conver) - 1):
            questions.append(diag[conver[i]])
            answers.append(diag[conver[i + 1]])

    short_input = []
    short_ans = []

    for i in range(len(questions)):
        if len(questions[i]) < 100 and len(answers[i]) < 100:
            short_ans.append(answers[i])
            short_input.append(questions[i])

    input_texts = []
    target_texts = []
    target_texts_inputs = []

    for line in short_input:
        input_texts.append(clean_text(line))
    for line in short_ans:
        target_texts.append(clean_text(line))

    for i in range(len(target_texts)):
        target_texts_inputs.append('<sos> ' + target_texts[i])
        target_texts[i] = target_texts[i] + ' <eos>'
        

    return input_texts[:1000], target_texts[:1000], target_texts_inputs[:1000]

input_texts, target_texts, target_texts_inputs = preprocessing()

embedding_matrix, word2idx_inputs, word2idx_outputs, num_words, max_out_len, max_input_len, num_words_output = embedding.embedder(input_texts, target_texts, target_texts_inputs)
encoder_input_sequences, decoder_input_sequences, decoder_targets_one_hot = embedding.embedder_data(input_texts, target_texts, target_texts_inputs,embedding_matrix, word2idx_inputs, word2idx_outputs, num_words, max_out_len, max_input_len, num_words_output)

model = load_model("seq2seq0.h5")


embedding_layer = Embedding(num_words, EMBEDDING_SIZE, weights=[embedding_matrix], input_length=max_input_len,trainable= False)

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

encoder_model = Model(encoder_inputs_placeholder, encoder_states)

decoder_state_input_h = Input(shape=(LSTM_NODES,))
decoder_state_input_c = Input(shape=(LSTM_NODES,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_inputs_single = Input(shape=(1,))
decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)
decoder_outputs, h, c = decoder_lstm(decoder_inputs_single_x, initial_state=decoder_states_inputs)

decoder_states = [h, c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(
    [decoder_inputs_single] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

idx2word_input = {v:k for k, v in word2idx_inputs.items()}
idx2word_target = {v:k for k, v in word2idx_outputs.items()}

def translate_sentence(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = word2idx_outputs['<sos>']
    eos = word2idx_outputs['<eos>']
    output_sentence = []

    for _ in range(max_out_len):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        idx = np.argmax(output_tokens[0, 0, :])

        if eos == idx:
            break

        word = ''

        if idx > 0:
            word = idx2word_target[idx]
            output_sentence.append(word)

        target_seq[0, 0] = idx
        states_value = [h, c]

    return ' '.join(output_sentence)
    
for i in range(10):
    i = np.random.choice(len(input_texts))
    input_seq = encoder_input_sequences[i:i+1]
    translation = translate_sentence(input_seq)
    print('-')
    print('Input:', input_texts[i])
    print('Response:', translation)