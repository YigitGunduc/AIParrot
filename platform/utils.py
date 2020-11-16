"""
utils.py returns :
    -characters
    -token_index
    -reverse_char_index

character set for model to train on
token index for one hot encoding our sentences
reverse_char_index for converting one hot encoded values to English
"""

import string
import re


def character_set():
    # creating a character set for model
    characters = sorted(list(re.sub(r'[A-Z]', '', string.printable)))
    # tokenizing every iten in characters set"
    token_index = dict([(char, i) for i, char in enumerate(characters)])
    # creating revers target index for translating models output to english
    reverse_char_index = dict((i, char) for char, i in token_index.items())
    return characters, token_index, reverse_char_index


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

