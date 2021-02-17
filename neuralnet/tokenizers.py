import pickle
import re
import tensorflow as tf

class Tokenizer:
    def __init__(self, maxlen=25, padding='post'):
        self.tokenizer = None
        self.maxlen = maxlen
        self.padding = padding


    def create_tokenizer(self, vocab_size=5000):
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token="<unk>", filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ',split=' ') 
        self.word_index = self.tokenizer.word_index
        self.index_word = self.tokenizer.index_word


    def tokenize_and_pad_training_data(self, encoder_input_data, decoder_input_data, decoder_output_data):
        text_corpus = encoder_input_data + decoder_input_data + decoder_output_data 
        self.tokenizer.fit_on_texts(text_corpus)

        self.word_index['<pad>'] = 0
        self.index_word[0] = '<pad>'

        encoder_input_data_tokenized = self.tokenizer.texts_to_sequences(encoder_input_data)
        decoder_input_data_tokenized = self.tokenizer.texts_to_sequences(decoder_input_data)
        decoder_output_data_tokenized = self.tokenizer.texts_to_sequences(decoder_output_data)


        encoder_input_data_padded = tf.keras.preprocessing.sequence.pad_sequences(encoder_input_data_tokenized, padding=self.padding, maxlen=self.maxlen)
        decoder_input_data_padded = tf.keras.preprocessing.sequence.pad_sequences(decoder_input_data_tokenized, padding=self.padding, maxlen=self.maxlen)
        decoder_output_data_padded = tf.keras.preprocessing.sequence.pad_sequences(decoder_output_data_tokenized, padding=self.padding, maxlen=self.maxlen)

        return encoder_input_data_padded, decoder_input_data_padded, decoder_output_data_padded


    def decode_sequence(self, encoded_text):
        lst = []
        for i in encoded_text:
            lst.append(self.tokenizer.index_word[i])
        return ' '.join(lst)


    def tokenize_sequence(self, sequence):
        tokenized_sequence = []
        sequence = sequence.lower()
        sequence = sequence.strip() 
        sequence = re.sub(r'[^\w\s]','',sequence)
        for i in sequence.split(' '):
            try:
                tokenized_sequence.append(self.tokenizer.word_index[i])
            except:
                tokenized_sequence.append(self.tokenizer.word_index['well'])
        if len(tokenized_sequence) > 25:
            tokenized_sequence = tokenized_sequence[:25]
        elif len(tokenized_sequence) == 25:
            tokenized_sequence = tokenized_sequence
        else:
            length = len(tokenized_sequence)
        for i in range(25-length):
            tokenized_sequence.append(0)
        return tokenized_sequence


    def save_tokenizer(self, name):
        with open(f'{name}.pickle', 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def load_tokenizer(self, path):
        with open(path, 'rb') as handle:
                tokenizer = pickle.load(handle)
                self.tokenizer = tokenizer
        return tokenizer


def create_tokenizer(vocab_size=5000):
    return tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token="<unk>", filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ',split=' ')


def tokenize_and_pad_training_data(tokenizer, encoder_input_data, decoder_input_data, decoder_output_data, maxlen=25, padding='post'):
    text_corpus = encoder_input_data + decoder_input_data + decoder_output_data 
    tokenizer.fit_on_texts(text_corpus)

    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    encoder_input_data_tokenized = tokenizer.texts_to_sequences(encoder_input_data)
    decoder_input_data_tokenized = tokenizer.texts_to_sequences(decoder_input_data)
    decoder_output_data_tokenized = tokenizer.texts_to_sequences(decoder_output_data)


    encoder_input_data_padded = tf.keras.preprocessing.sequence.pad_sequences(encoder_input_data_tokenized, padding=padding, maxlen=maxlen)
    decoder_input_data_padded = tf.keras.preprocessing.sequence.pad_sequences(decoder_input_data_tokenized, padding=padding, maxlen=maxlen)
    decoder_output_data_padded = tf.keras.preprocessing.sequence.pad_sequences(decoder_output_data_tokenized, padding=padding, maxlen=maxlen)

    return encoder_input_data_padded, decoder_input_data_padded, decoder_output_data_padded


def decode_sequence(tokenizer, encoded_text):
    lst = []
    for i in encoded_text:
        lst.append(tokenizer.index_word[i])
    return ' '.join(lst)


def tokenize_sequence(tokenizer, sequence):
    tokenized_sequence = []
    sequence = sequence.lower()
    sequence = sequence.strip() 
    sequence = re.sub(r'[^\w\s]','',sequence)
    for i in sequence.split(' '):
        try:
            tokenized_sequence.append(tokenizer.word_index[i])
        except:
            tokenized_sequence.append(tokenizer.word_index['well'])
    if len(tokenized_sequence) > 25:
        tokenized_sequence = tokenized_sequence[:25]
    elif len(tokenized_sequence) == 25:
        tokenized_sequence = tokenized_sequence
    else:
        length = len(tokenized_sequence)
    for i in range(25-length):
        tokenized_sequence.append(0)
    return tokenized_sequence


def save_tokenizer(tokenizer, name):
    with open(f'{name}.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_tokenizer(path):
    with open(path, 'rb') as handle:
            tokenizer = pickle.load(handle)
    return tokenizer
