import re


def preprocessing():

    lines = open("../data/movie_lines.txt", encoding="utf-8", errors="ignore").read().split("\n")
    convers = open("../data/movie_conversations.txt", encoding="utf-8", errors="ignore").read().split("\n")

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

    for line in short_input:
        input_texts.append(clean_text(line))
    for line in short_ans:
        target_texts.append(clean_text(line))

    for i in range(len(target_texts)):
        target_texts[i] = '\t' + target_texts[i] + '\n'

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])
    return input_texts, target_texts, max_encoder_seq_length, max_decoder_seq_length
