import tensorflow as tf
import utils
from preprocessing import create_training_data
from tokenizers import Tokenizer
from model import seq2seq
from config import (VOCAB_SIZE,
                    MAXLEN,
                    EPOCHS,
                    SAVE_AT,  
                    LEARNING_RATE,
                    BATCH_SIZE,
                    VERBOSE,
                    LOSS)

tokenizer = Tokenizer()

encoder_input_data, decoder_input_data, decoder_output_data = create_training_data()  # parsing the dataset and creating conversation pairs

encoder_input_data, decoder_input_data, decoder_output_data  = tokenizer.tokenize_and_pad_training_data(encoder_input_data, decoder_input_data, decoder_output_data)  # tokenizing and padding those pairs

tokenizer.save_tokenizer(f'tokenizer-vocab_size-{VOCAB_SIZE}')  # saving tokenizer for layer use

Seq2SeqModel = seq2seq()  # creating the seq2seq model

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0, clipvalue=0.5)
Seq2SeqModel.compile(optimizer=optimizer, loss=LOSS, metrics=['accuracy'])
Seq2SeqModel.summary()

def train(model, encoder_input_data, decoder_input_data, decoder_output_data, epochs, batch_size, verbose, save_at):
    with tf.device('/device:GPU:0' if utils.check_cuda else '/cpu:0'):
        for epoch in range(1, epochs+1):
            print(f'Epochs {epoch}/{epochs}')
            model.fit([encoder_input_data, decoder_input_data], decoder_output_data, batch_size=batch_size, epochs=1, verbose=verbose)
            if epoch % SAVE_AT == 0:
                model.save_weights(f'seq2seq-weights-{epochs}-epochs-{LEARNING_RATE}-learning_rate.h5')


train(Seq2SeqModel, encoder_input_data, decoder_input_data, decoder_output_data, EPOCHS, BATCH_SIZE, VERBOSE, SAVE_AT)

