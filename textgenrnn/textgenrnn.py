from keras.layers import Input, Embedding, Dense, LSTM
from keras.callbacks import LearningRateScheduler
from keras.models import Model, load_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import numpy as np
import json
import h5py
from pkg_resources import resource_filename


class textgenrnn:
    META_TOKEN = '<s>'

    def __init__(self, weights_path=None,
                 vocab_path=None):

        if weights_path is None:
            weights_path = resource_filename(__name__,
                                             'textgenrnn_weights.hdf5')

        if vocab_path is None:
            vocab_path = resource_filename(__name__,
                                           'textgenrnn_vocab.json')

        with open(vocab_path, 'r') as json_file:
            self.vocab = json.load(json_file)

        self.tokenizer = Tokenizer(filters='', char_level=True)
        self.tokenizer.word_index = self.vocab
        self.num_classes = len(self.vocab) + 1
        self.model = textgenrnn_model(weights_path, self.num_classes)
        self.indices_char = dict((self.vocab[c], c) for c in self.vocab)

    def generate(self, n=1, return_as_list=False, **kwargs):
        gen_texts = []
        for _ in range(n):
            gen_text = textgenrnn_generate(self.model,
                                           self.vocab,
                                           self.indices_char,
                                           **kwargs)
            if not return_as_list:
                print("{}\n".format(gen_text))
            gen_texts.append(gen_text)
        if return_as_list:
            return gen_texts

    def train_on_texts(self, texts, batch_size=128, num_epochs=50, verbose=1):

        # Encode chars as X and y.
        X = []
        y = []

        for text in texts:
            subset_x, subset_y = textgenrnn_encode_training(text,
                                                            self.META_TOKEN)
            for i in range(len(subset_x)):
                X.append(subset_x[i])
                y.append(subset_y[i])

        X = np.array(X)
        y = np.array(y)

        X = self.tokenizer.texts_to_sequences(X)
        X = sequence.pad_sequences(X, maxlen=40)
        y = textgenrnn_encode_cat(y, self.vocab)

        base_lr = 2e-3

        # scheduler function must be defined inline.
        def lr_linear_decay(epoch):
            return (base_lr * (1 - (epoch / num_epochs)))

        self.model.fit(X, y, batch_size=batch_size, epochs=num_epochs,
                       callbacks=[LearningRateScheduler(lr_linear_decay)],
                       verbose=verbose)

    def save(self, weights_path="textgenrnn_weights_saved.hdf5"):
        self.model.save_weights(weights_path)

    def load(self, weights_path):
        self.model = textgenrnn_model(weights_path, self.num_classes)

    def reset(self):
        self.model = textgenrnn_model(
            resource_filename(__name__,
                              'textgenrnn_weights.hdf5'),
            self.num_classes)

    def train_from_file(self, file_path, header=True, delim="\n", **kwargs):
        texts = []
        texts = textgenrnn_texts_from_file(file_path, header, delim)
        print("{} texts collected.".format(len(texts)))
        self.train_on_texts(texts, **kwargs)

    def train_from_largetext_file(self, file_path, **kwargs):
        self.train_from_file(file_path, delim="\n\n", **kwargs)

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
                         trainable=True, name='embedding')(input)
    rnn = LSTM(128, return_sequences=False, name='rnn')(embedded)
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


def textgenrnn_generate(model, vocab,
                        indices_char, prefix=None, temperature=0.2,
                        maxlen=40, meta_token='<s>',
                        max_gen_length=200):
    '''
    Generates and returns a single text.
    '''

    text = [meta_token] + list(prefix) if prefix else [meta_token]
    next_char = ''

    while next_char != meta_token and len(text) < max_gen_length:
        encoded_text = textgenrnn_encode_sequence(text[-maxlen:],
                                                  vocab, maxlen)
        next_index = textgenrnn_sample(
            model.predict(encoded_text, batch_size=1)[0],
            temperature)
        next_char = indices_char[next_index]
        text += [next_char]
    return ''.join(text[1:-1])


def textgenrnn_encode_sequence(text, vocab, maxlen):
    '''
    Encodes a text into the corresponding encoding for prediction with
    the model.
    '''

    encoded = np.array([vocab.get(x, 0) for x in text])
    return sequence.pad_sequences([encoded], maxlen=maxlen)


def textgenrnn_encode_training(text, meta_token='<s>', maxlen=40):
    '''
    Encodes a list of texts into a list of texts, and the next character
    in those texts.
    '''

    text_aug = [meta_token] + list(text) + [meta_token]
    chars = []
    next_char = []

    for i in range(len(text_aug) - 1):
        chars.append(text_aug[0:i + 1][-maxlen:])
        next_char.append(text_aug[i + 1])

    return chars, next_char


def textgenrnn_texts_from_file(file_path, header=True, delim='\n'):
    '''
    Retrieves texts from a newline-delimited file and returns as a list.
    '''

    with open(file_path, 'r', encoding="utf-8") as f:
        if header:
            f.readline()
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
