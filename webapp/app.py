from flask import Flask, render_template, request
from tokenizers import Tokenizer
from generate import Predict, Seq2SeqModel

tokenizer = Tokenizer()

tokenizer.load_tokenizer('tokenizer-vocab_size-5000.pickle')

# loading pretrained weight
Seq2SeqModel.load_weights('seq2seq-weigths-epochs-1100.h5')

predict = Predict(Seq2SeqModel, tokenizer)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    try:
        answer = str(predict.create_response(str(userText)))
    except:
        answer = "I couldn't quite get it."

    return str(predict.create_response(str(userText)))


if __name__ == "__main__":
    app.run(debug=True)

