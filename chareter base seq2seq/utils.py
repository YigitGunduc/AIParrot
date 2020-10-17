"""
utils.py returns :
    -characters
    -token_index
    -reverse_char_index

character set for model to train on
token index for one hot encoding our sentences
reverse_char_index for converting one hot encoden valuest to english
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
