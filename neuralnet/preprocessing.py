import re


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


def create_training_data():
    
    lines = open("../data/movie_lines.txt", encoding="utf-8", errors="ignore").read().split("\n")
    convers = open("../data/movie_conversations.txt", encoding="utf-8", errors="ignore").read().split("\n")

    exchn = []
    for conver in convers:
        exchn.append(conver.split(" +++$+++ ")[-1][1:-1].replace("'", "").replace(",", "").split())

    diag = {}
    for line in lines:
        diag[line.split(" +++$+++ ")[0]] = line.split(" +++$+++ ")[-1]

    encoder_input_data = []
    decoder_input_data = []
    decoder_output_data = []

    for conver in exchn:
        for i in range(len(conver) - 1):

            encoder_input_data.append(clean_text(diag[conver[i]]))
            decoder_input_data.append('<sos> ' + clean_text(diag[conver[i + 1]]))
            decoder_output_data.append(clean_text(diag[conver[i + 1]]) + ' <eos>')


    return encoder_input_data, decoder_input_data, decoder_output_data

