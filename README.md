
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OAg9u-dQ1jofqUK2KT9DaxpqLl_8SOuG?usp=sharing)


<!-- PROJECT LOGO -->
<br />
<p align="center">
  <img src="https://img.icons8.com/plasticine/100/000000/parrot.png"/>
  <h1 align="center">AIParrot</h1>
    
  <p align="center">
    AIParrot is an intelligent conversational AI that uses sequence models and word embedding to generate responses to a given question. <br />
   <br />
    <a href="https://colab.research.google.com/drive/13k-AfkOVw_8zcKEsrr-dRNLX0E9M4wOa?usp=sharing">Open In Colab</a>
  </p>
</p>


### Built With

This project is built using Python and Tensorflow & Keras.

* [Python](https://www.python.org/)
* [TensorFlow](https://www.tensorflow.org/)
* [Keras](https://keras.io/)

## Model
Sequence-to-sequence(Seq2Seq) models are built for converting sequences from one domain to a meaningful sequence in another domain (e.g. machine translation, chatbot...).

![seq2seq model](https://pytorch.org/tutorials/_images/seq2seq_ts.png)



<!-- GETTING STARTED -->
## Getting Started

### Installation
```sh
# clone the repo
git clone https://github.com/YigitGunduc/AIParrot.git

# install requirements
pip install -r requirements.txt
```

### Training

```sh
# navigate to the AIParrot/neuralnet folder 
cd AIParrot/neuralnet

#run train.py 
python3 train.py
```
to tweak model params and values see AIParrot/neuralnet/config.py

### Generating Text from Trained Model

```sh
python3 chat.py
```
or
```python
from tokenizers import Tokenizer
from generate import Predict, Seq2SeqModel

tokenizer = Tokenizer()

# loading tokenizer
tokenizer.load_tokenizer('AIParrot/neuralnet/tokenizer-vocab_size-5000.pickle')

# loading pretrained weight
Seq2SeqModel.load_weights('AIParrot/weights/seq2seq-weigths-epochs-1100.h5')

predict = Predict(Seq2SeqModel, tokenizer)

print(predict.create_response('How are you?'))
```

### Running the Web-App Locally

```sh
# navigate to the AIParrot/webapp folder 
cd AIParrot/webapp

# run app.py
python3 app.py

# check out http://127.0.0.1:5000/
```

### API
AIParrots web API can used as shown below

```python
import requests 

response = requests.get("/http://127.0.0.1:5000/api/?q=QUERY")

#raw response
print(response.json())

#cleaned up response
print(response.json()["response"])

'''
{
  "response": "hey"
}
'''
```

<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/YigitGunduc/Spectrum/issues) for a list of proposed features (and known issues).


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



[contributors-shield]: https://img.shields.io/github/contributors/YigitGunduc/AIParrot.svg?style=flat-rounded
[contributors-url]: https://github.com/YigitGunduc/AIParrot/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/YigitGunduc/AIParrot.svg?style=flat-rounded
[forks-url]: https://github.com/YigitGunduc/AIParrot/network/members
[stars-shield]: https://img.shields.io/github/stars/YigitGunduc/AIParrot.svg?style=flat-rounded
[stars-url]: https://github.com/YigitGunduc/AIParrot/stargazers
[issues-shield]: https://img.shields.io/github/issues/YigitGunduc/AIParrot.svg?style=flat-rounded
[issues-url]: https://github.com/YigitGunduc/AIParrot/issues
[license-url]: https://github.com/YigitGunduc/AIParrot/blob/master/LICENSE
[license-shield]: https://img.shields.io/github/license/YigitGunduc/AIParrot.svg?style=flat-rounded
