import numpy as np
from numpy import asarray
from numpy import zeros
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

BATCH_SIZE = 64
EPOCHS = 100
LSTM_NODES = 256
NUM_SENTENCES = 20000
MAX_SENTENCE_LENGTH = 25
MAX_NUM_WORDS = 11000
EMBEDDING_SIZE = 100


def embedder(input_texts, target_texts, target_texts_inputs):
    input_tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    input_tokenizer.fit_on_texts(input_texts)
    input_integer_seq = input_tokenizer.texts_to_sequences(input_texts)

    word2idx_inputs = input_tokenizer.word_index

    print('Total unique words in the input: %s' % len(word2idx_inputs))

    max_input_len = max(len(sen) for sen in input_integer_seq)
    print("Length of longest sentence in input: %g" % max_input_len)

    output_tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, filters='')
    output_tokenizer.fit_on_texts(target_texts + target_texts_inputs)
    output_integer_seq = output_tokenizer.texts_to_sequences(target_texts)
    output_input_integer_seq = output_tokenizer.texts_to_sequences(target_texts_inputs)

    word2idx_outputs = output_tokenizer.word_index
    print('Total unique words in the output: %s' % len(word2idx_outputs))

    num_words_output = len(word2idx_outputs) + 1
    max_out_len = max(len(sen) for sen in output_integer_seq)
    print("Length of longest sentence in the output: %g" % max_out_len)

    print(word2idx_inputs["do"])
    print(word2idx_inputs["what"])

    embeddings_dictionary = dict()

    glove_file = open(r'../../glove/glove.6B.100d.txt', encoding="utf8")

    for line in glove_file:
        records = line.split()
        word = records[0]
        vector_dimensions = asarray(records[1:], dtype='float32')
        embeddings_dictionary[word] = vector_dimensions
    glove_file.close()

    num_words = min(MAX_NUM_WORDS, len(word2idx_inputs) + 1)
    embedding_matrix = zeros((num_words, EMBEDDING_SIZE))

    for word, i in word2idx_inputs.items():
        if i < num_words:
            emb_vec = embeddings_dictionary.get(word)
            if emb_vec is not None:
                embedding_matrix[i] = emb_vec

    return embedding_matrix, word2idx_inputs, word2idx_outputs, num_words, max_out_len, max_input_len, num_words_output


def embedder_data(input_texts, target_texts, target_texts_inputs, embedding_matrix, word2idx_inputs, word2idx_outputs, num_words, max_out_len, max_input_len, num_words_output):
    input_tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    input_tokenizer.fit_on_texts(input_texts)
    input_integer_seq = input_tokenizer.texts_to_sequences(input_texts)

    output_tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, filters='')
    output_tokenizer.fit_on_texts(target_texts + target_texts_inputs)
    output_integer_seq = output_tokenizer.texts_to_sequences(target_texts)
    output_input_integer_seq = output_tokenizer.texts_to_sequences(target_texts_inputs)

    encoder_input_sequences = pad_sequences(input_integer_seq, maxlen=max_input_len, padding='post')

    decoder_input_sequences = pad_sequences(output_input_integer_seq, maxlen=max_out_len, padding='post')

    decoder_targets_one_hot = np.zeros((len(input_texts), max_out_len, num_words_output), dtype='float32')
    print("--------------------------------------------------")
    print("====>",len(input_texts), max_out_len, num_words_output)
    for i, d in enumerate(output_input_integer_seq):
        for t, word in enumerate(d):
            decoder_targets_one_hot[i, t, word] = 1

    return encoder_input_sequences, decoder_input_sequences, decoder_targets_one_hot
