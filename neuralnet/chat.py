from tokenizers import Tokenizer
from generate import Predict, Seq2SeqModel

tokenizer = Tokenizer()

# loading tokenizer
tokenizer.load_tokenizer('tokenizer-vocab_size-5000.pickle')

# loading pretrained weight
Seq2SeqModel.load_weights('../weights/seq2seq-weigths-epochs-1100.h5')

predict = Predict(Seq2SeqModel, tokenizer)

print(predict.create_response('How are you?'))

