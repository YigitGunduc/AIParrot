import re


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

    input_texts = []
    target_texts = []

    for line in questions:
        input_texts.append(clean_text(line))
    for line in answers:
        target_texts.append(clean_text(line))

    return input_texts, target_texts


def futherprocessing():

    input_texts = []
    target_texts = []

    f_input = open("..\\data\\input_texts_cleaned.txt", "r")
    inp = f_input.read().split("\n")

    f_target = open("..\\data\\target_texts_cleaned.txt", "r")
    target = f_target.read().split("\n")

    for i in range(len(inp)):
        input_texts.append(inp[i])
        target_texts.append(target[i])

    target_texts_inputs = []

    for i in range(len(target_texts)):
        target_texts_inputs.append('<sos> ' + target_texts[i])
        target_texts[i] = target_texts[i] + ' <eos>'

    return input_texts, target_texts, target_texts_inputs
