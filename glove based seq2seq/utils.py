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
import numpy as np

def character_set():
    # creating a character set for model
    characters = sorted(list(re.sub(r'[A-Z]', '', string.printable)))
    # tokenizing every iten in characters set"
    token_index = dict([(char, i) for i, char in enumerate(characters)])
    # creating revers target index for translating models output to english
    reverse_char_index = dict((i, char) for char, i in token_index.items())
    return characters, token_index, reverse_char_index


class Preparedata():


    def __init__(self,DIGITS):
        self.CharacterSet = sorted(list(re.sub(r'[A-Z]', '', string.printable)))
        self.MaxLength = len(self.CharacterSet)
        self.DIGITS = DIGITS
        self.Dict = dict()
        self.__Dictionary()
        
    def __Dictionary(self):
        for ind,content in enumerate(sorted(list(self.CharacterSet))):
            self.Dict[ind] = content
            self.Dict[content] = ind
        return self.Dict
    
    def Encoder(self,InputString,Length):
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
        return "".join(Values)
