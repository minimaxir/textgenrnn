from keras.layers import Input, Embedding, Dense, GRU
from keras.optimizers import Nadam
from keras.callbacks import ModelCheckpoint, Callback
from keras.callbacks import LearningRateScheduler, LambdaCallback
from keras.models import Model, load_model
from keras.preprocessing import sequence
from random import random
import numpy as np
import json
import h5py
import csv


class textgenrnn():
    META_TOKEN = '<s>'

    def __init__(self, weights_path, vocab_path):
        with open(vocab_path, 'r') as json_file:
            self.vocab = json.load(json_file)

        self.tokenizer = Tokenizer()
        self.tokenizer.word_index = self.vocab
        self.num_classes = len(self.tokenizer.word_index) + 1
        self.model = textgenrnn_model(weights_path, self.num_classes)
        self.indices_char = dict((self.vocab[c], c) for c in self.vocab)

    def generate(self, n=1, return_as_list=False, **kwargs):
        gen_texts = []
        for _ in range(n):
            gen_text = textgenrnn_generate(model, **kwargs)
            print(gen_text)
            gen_texts.append(gen_text)
        if return_as_list:
            return gen_texts

    def train_on_texts(self, texts, batch_size=128, num_epochs=50):

        # Encode chars as X and y.
        X = []
        y = []

        for text in texts:
            subset_x, subset_y = textgenrnn_encode_training(text, meta_token)
            for i in range(len(subset_x)):
                if random() < 0.33:
                    X.append(subset_x[i])
                    y.append(subset_y[i])

        X = np.array(X)
        y = np.array(y)

        X = self.tokenizer.texts_to_sequences(X)
        X = sequence.pad_sequences(X, maxlen=MAX_LENGTH)
        y = textgenrnn_encode_cat(y, self.tokenizer.word_index)

        base_lr = 2e-3

        # scheduler function must be defined inline.
        def lr_linear_decay(epoch):
            return (base_lr * (1 - (epoch / num_epochs)))

        self.model.fit(X, y, batch_size=batch_size, epochs=num_epochs,
                       callbacks=[LearningRateScheduler(lr_linear_decay)])

    def train_from_file(self, file_path, **kwargs):
        files = [file_path] if file_path is str else file_path
        texts = []
        for file in files:
            texts += textgenrnn_texts_from_file(file, **kwargs)
        self.train_on_texts(texts, **kwargs)

    def generate_to_file(self, destination_path, **kwargs):
        texts = self.generate(**kwargs, return_as_list=True)
        with open(destination_path, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write("{}\n".format(text))


def textgenrnn_model(weights_path, num_classes, maxlen=40):
    '''
    Builds the model architecture for textgenrnn and
    loads the pretrained weights for the model.
    '''

    input = Input(shape=(maxlen,), name='input')
    embedded = Embedding(num_classes, 100, input_length=maxlen,
                         training=False, name='embedding')(input)
    rnn = GRU(128, return_sequences=False, name='rnn')(embedded)
    output = Dense(num_classes, name='output', activation='softmax')(rnn)

    model = Model(inputs=[input], outputs=[output])
    model.load_weights(weights_path, by_name=True)
    model.compile(loss='categorical_crossentropy', optimizer='nadam')
    return model


def textgenrnn_sample(preds, temperature):
    '''
    Samples predicted probabilities of the next character to allow
    for the network to show "creativity."
    '''

    preds = np.asarray(preds).astype('float64')

    if temperature is None or temperature == 0.0:
        return np.argmax(preds)

    preds = np.log(preds + 1e-12) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    index = -1

    # prevent function from being able to choose 0 (placeholder)
    while index < 1:
        probas = np.random.multinomial(1, preds, 1)
        index = np.argmax(probas)
    return index


def textgenrnn_generate(model, prefix=None, temperature=1.0,
                        maxlen=40, meta_token='<s>',
                        max_gen_length=300):
    '''
    Generates and returns a single text.
    '''

    text = [meta_token] + list(prefix) if prefix else [meta_token]
    next_char = ''

    while next_char != meta_token and len(text) < max_gen_length:
        encoded_text = encode_sequence(text[-maxlen:])
        next_index = textgenrnn_sample(
            model.predict(encoded_text, batch_size=1)[0])
        next_char = indices_char[next_index]
        text += [next_char]
    return ''.join(text[1:-1])


def textgenrnn_encode_sequence(text, vocab):
    '''
    Encodes a string into the corresponding encoding for prediction with
    the model.
    '''

    encoded = np.array([vocab.get(x, 0) for x in text])
    return sequence.pad_sequences(encoded, maxlen=maxlen)


def textgenrnn_encode_training(text, meta_token='<s>', maxlen=40):
    '''
    Encodes a list of texts into a sequence of texts, and the next character
    in those texts.
    '''

    text_aug = [meta_token] + list(text) + [meta_token]
    chars = []
    next_char = []

    for i in range(len(text_aug) - 1):
        chars.append(text_aug[0:i + 1][-maxlen:])
        next_char.append(text_aug[i + 1])

    return chars, next_char


def textgenrnn_texts_from_file(file_path, header=True):
    '''
    Retrieves texts from a newline-delimited file and returns as a list.
    '''

    with open(file_path, 'r', encoding="utf-8", delim='\n') as f:
        f.readline() if header
        texts = [line.rstrip(delim) for line in f]
    return texts


def textgenrnn_encode_cat(chars, vocab):
    '''
    One-hot encodes values at given chars efficiently by preallocating
    a zeros matrix.
    '''

    a = np.float32(np.zeros((len(chars), len(vocab) + 1)))
    rows, cols = zip(*[(i, vocab.get(char, 0))
                       for i, char in enumerate(chars)])
    a[rows, cols] = 1
    return a
