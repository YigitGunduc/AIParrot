#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import re
import sys
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


# In[99]:


BATCH_SIZE = 64
EPOCHS = 100
LSTM_NODES =256
NUM_SENTENCES = 20000
MAX_SENTENCE_LENGTH = 25
MAX_NUM_WORDS = 11000
EMBEDDING_SIZE = 100


# In[97]:


def preprocessing():

    lines = open("C:/Users/gunduc/Desktop/parrot/data/movie_lines.txt", encoding="utf-8", errors="ignore").read().split("\n")
    convers = open("C:/Users/gunduc/Desktop/parrot/data/movie_conversations.txt", encoding="utf-8", errors="ignore").read().split("\n")

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
        

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])
    return input_texts[:30000], target_texts[:30000], target_texts_inputs[:30000], max_encoder_seq_length, max_decoder_seq_length


# In[98]:


input_texts, target_texts, target_texts_inputs, _, _ = preprocessing()


# In[100]:


print(input_texts[172])
print(target_texts[172])
print(target_texts_inputs[172])


# In[101]:


input_tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
input_tokenizer.fit_on_texts(input_texts)
input_integer_seq = input_tokenizer.texts_to_sequences(input_texts)

word2idx_inputs = input_tokenizer.word_index

print('Total unique words in the input: %s' % len(word2idx_inputs))

max_input_len = max(len(sen) for sen in input_integer_seq)
print("Length of longest sentence in input: %g" % max_input_len)


# In[102]:


output_tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, filters='')
output_tokenizer.fit_on_texts(target_texts + target_texts_inputs)
output_integer_seq = output_tokenizer.texts_to_sequences(target_texts)
output_input_integer_seq = output_tokenizer.texts_to_sequences(target_texts_inputs)

word2idx_outputs = output_tokenizer.word_index
print('Total unique words in the output: %s' % len(word2idx_outputs))

num_words_output = len(word2idx_outputs) + 1
max_out_len = max(len(sen) for sen in output_integer_seq)
print("Length of longest sentence in the output: %g" % max_out_len)


# In[103]:


encoder_input_sequences = pad_sequences(input_integer_seq, maxlen=max_input_len, padding='post')
print("encoder_input_sequences.shape:", encoder_input_sequences.shape)
print("encoder_input_sequences[172]:", encoder_input_sequences[172])


# In[104]:


print(word2idx_inputs["do"])
print(word2idx_inputs["what"])


# In[105]:


decoder_input_sequences = pad_sequences(output_input_integer_seq, maxlen=max_out_len, padding='post')
print("decoder_input_sequences.shape:", decoder_input_sequences.shape)
print("decoder_input_sequences[172]:", decoder_input_sequences[172])


# In[106]:


print(word2idx_outputs["<sos>"])
print(word2idx_outputs["me"])
print(word2idx_outputs["you"])
print(word2idx_outputs["hello."])


# In[107]:


from numpy import array
from numpy import asarray
from numpy import zeros

embeddings_dictionary = dict()

glove_file = open(r'C:/Users/gunduc/Desktop/parrot/data/glove.6B.100d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions
glove_file.close()


# In[108]:


num_words = min(MAX_NUM_WORDS, len(word2idx_inputs) + 1)
embedding_matrix = zeros((num_words, EMBEDDING_SIZE))

for word, i in word2idx_inputs.items():
    if i < num_words:
        emb_vec = embeddings_dictionary.get(word)
        if emb_vec is not None:
            embedding_matrix[i] = emb_vec


# In[109]:


embedding_matrix


# In[110]:


print(embeddings_dictionary["ill"])


# In[111]:


embedding_layer = Embedding(num_words, EMBEDDING_SIZE, weights=[embedding_matrix], input_length=max_input_len,trainable= False)


# In[112]:


decoder_targets_one_hot = np.zeros((
        len(input_texts),
        max_out_len,
        num_words_output
    ),
    dtype='float32'
)


# In[113]:


max_out_len


# In[114]:


decoder_targets_one_hot.shape


# In[115]:


for i, d in enumerate(output_input_integer_seq):
    for t, word in enumerate(d):
        decoder_targets_one_hot[i, t, word] = 1


# In[116]:


encoder_inputs_placeholder = Input(shape=(max_input_len,))
x = embedding_layer(encoder_inputs_placeholder)
encoder = LSTM(LSTM_NODES, return_state=True)
encoder_outputs, h, c = encoder(x)
encoder_states = [h, c]


# In[117]:


decoder_inputs_placeholder = Input(shape=(max_out_len,))
decoder_embedding = Embedding(num_words_output, LSTM_NODES)


# In[118]:


decoder_inputs_x = decoder_embedding(decoder_inputs_placeholder)
decoder_lstm = LSTM(LSTM_NODES, return_sequences=True, return_state=True)


# In[119]:


decoder_outputs, _, _ = decoder_lstm(decoder_inputs_x, initial_state=encoder_states)
decoder_dense = Dense(num_words_output, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)


# In[120]:


model = Model([encoder_inputs_placeholder, decoder_inputs_placeholder], decoder_outputs)  


# In[121]:


decoder_outputs.shape


# In[122]:


decoder_inputs.shape


# In[123]:


num_words_output


# In[124]:


model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


# In[125]:


r = model.fit(
    [encoder_input_sequences, decoder_input_sequences],
    decoder_targets_one_hot,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.1,
)


# In[ ]:





# In[ ]:





# In[82]:


encoder_model = Model(encoder_inputs_placeholder, encoder_states)


# In[83]:


decoder_state_input_h = Input(shape=(LSTM_NODES,))
decoder_state_input_c = Input(shape=(LSTM_NODES,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]


# In[84]:


decoder_inputs_single = Input(shape=(1,))
decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)


# In[85]:


decoder_outputs, h, c = decoder_lstm(decoder_inputs_single_x, initial_state=decoder_states_inputs)


# In[86]:


decoder_states = [h, c]
decoder_outputs = decoder_dense(decoder_outputs)


# In[87]:


decoder_model = Model(
    [decoder_inputs_single] + decoder_states_inputs,
    [decoder_outputs] + decoder_states
)


# In[88]:


idx2word_input = {v:k for k, v in word2idx_inputs.items()}
idx2word_target = {v:k for k, v in word2idx_outputs.items()}


# In[89]:


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


# In[96]:


i = np.random.choice(len(input_texts))
input_seq = encoder_input_sequences[i:i+1]
translation = translate_sentence(input_seq)
print('-')
print('Input:', input_texts[i])
print('Response:', translation)


# In[ ]:





# In[ ]:




