import re
import string
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
import numpy as np

latent_dim = 256  # Latent dimensionality of the encoding space.

class Preparedata():

    def __init__(self,CharacterSet):
        self.MaxLength = len(CharacterSet)
        self.CharacterSet = CharacterSet
        self.Dict = dict()
        self.__Dictionary()
        
    def __Dictionary(self):
        for ind,content in enumerate(sorted(list(self.CharacterSet))):
            self.Dict[ind] = content
            self.Dict[content] = ind
        return self.Dict
    
    def Encoder(self,InputString,Length = 1):
        NN = len(InputString)
        Encoded = np.zeros((NN,Length,len(self.CharacterSet)))
        for n in range(NN):
            for ind,content in enumerate(InputString[n]):
                Encoded[n,ind,self.Dict[content]] = 1
        return Encoded   
    
    def Decoder(self,EncodedText):
        N = EncodedText.shape
        NN = N[0]
        MM = N[1]
        KK = N[2]
        XX = np.zeros(KK,dtype=int)
        XX[self.Dict[' ']] = 1
        Values = [] 
        print(NN,MM,XX)
        sxx = np.sum(EncodedText,axis=2)
        ind = np.where(sxx == 0)
        EncodedText[ind] = XX
        for n in range(NN):    
            Str = ""
            for ind,content in enumerate(EncodedText[n]):
                Str += "".join(self.Dict[np.argmax(content)])
            Values.append(Str)    
        return Values

def clean_text(txt):
    txt = txt.lower()
    txt = re.sub(r"i'm", "i am", txt)
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


input_characters = sorted(list(re.sub(r'[A-Z]', '', string.printable)))
target_characters = sorted(list(re.sub(r'[A-Z]', '', string.printable)))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)


print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)

input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])

# Restore the model and construct the encoder and decoder.
model = load_model('seq2seq0.h5')

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
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


# Decodes an input sequence.  Future work should support beam search.
def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence

prep = Preparedata(input_characters)


input_seq = prep.Encoder("hello",1)
decoded_sentence = decode_sequence(input_seq)
print('-')
print('Input sentence:', input_seq)
print('Decoded sentence:', decoded_sentence)
