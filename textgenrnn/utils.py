from keras.callbacks import LearningRateScheduler, Callback
from keras.models import Model, load_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras import backend as K
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import json
import h5py
import csv
import re


def textgenrnn_sample(preds, temperature):
    '''
    Samples predicted probabilities of the next character to allow
    for the network to show "creativity."
    '''

    preds = np.asarray(preds).astype('float64')

    if temperature is None or temperature == 0.0:
        return np.argmax(preds)

    preds = np.log(preds + K.epsilon()) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    index = np.argmax(probas)

    # prevent function from being able to choose 0 (placeholder)
    # choose 2nd best index from preds
    if index == 0:
        index = np.argsort(preds)[-2]

    return index


def textgenrnn_generate(model, vocab,
                        indices_char, prefix=None, temperature=0.5,
                        maxlen=40, meta_token='<s>',
                        word_level=False,
                        single_text=False,
                        max_gen_length=300):
    '''
    Generates and returns a single text.
    '''

    if single_text:
        text = list(prefix) if prefix else ['']
        max_gen_length += maxlen
    else:
        text = [meta_token] + list(prefix) if prefix else [meta_token]
    next_char = ''

    if model_input_count(model) > 1:
        model = Model(inputs=model.input[0], outputs=model.output[1])

    while next_char != meta_token and len(text) < max_gen_length:
        encoded_text = textgenrnn_encode_sequence(text[-maxlen:],
                                                  vocab, maxlen)
        next_index = textgenrnn_sample(
            model.predict(encoded_text, batch_size=1)[0],
            temperature)
        next_char = indices_char[next_index]
        text += [next_char]

    collapse_char = ' ' if word_level else ''

    # if single text, ignore sequences generated w/ padding
    # if not single text, strip the <s> meta_tokens
    if single_text:
        text = text[maxlen:]
    else:
        text = text[1:-1]

    text_joined = collapse_char.join(text)

    # If word level, remove spaces around punctuation for cleanliness.
    if word_level:
        #     left_punct = "!%),.:;?@]_}\\n\\t'"
        #     right_punct = "$([_\\n\\t'"
        punct = '\\n\\t'
        text_joined = re.sub(" ([{}]) ".format(punct), r'\1', text_joined)
        #     text_joined = re.sub(" ([{}])".format(
        #       left_punct), r'\1', text_joined)
        #     text_joined = re.sub("([{}]) ".format(
        #       right_punct), r'\1', text_joined)

    return text_joined


def textgenrnn_encode_sequence(text, vocab, maxlen):
    '''
    Encodes a text into the corresponding encoding for prediction with
    the model.
    '''

    encoded = np.array([vocab.get(x, 0) for x in text])
    return sequence.pad_sequences([encoded], maxlen=maxlen)


def textgenrnn_texts_from_file(file_path, header=True, delim='\n'):
    '''
    Retrieves texts from a newline-delimited file and returns as a list.
    '''

    with open(file_path, 'r', encoding='utf8', errors='ignore') as f:
        if header:
            f.readline()
        texts = [line.rstrip(delim).strip('"') for line in f]

    return texts


def textgenrnn_texts_from_file_context(file_path, header=True):
    '''
    Retrieves texts+context from a two-column CSV.
    '''

    with open(file_path, 'r', encoding='utf8', errors='ignore') as f:
        if header:
            f.readline()
        texts = []
        context_labels = []
        reader = csv.reader(f)
        for row in reader:
            texts.append(row[0])
            context_labels.append(row[1])

    return (texts, context_labels)


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


def model_input_count(model):
    if isinstance(model.input, list):
        return len(model.input)
    else:   # is a Tensor
        return model.input.shape[0]


class generate_after_epoch(Callback):
    def __init__(self, textgenrnn, gen_epochs, max_gen_length):
        self.textgenrnn = textgenrnn
        self.gen_epochs = gen_epochs
        self.max_gen_length = max_gen_length

    def on_epoch_end(self, epoch, logs={}):
        if self.gen_epochs > 0 and (epoch+1) % self.gen_epochs == 0:
            self.textgenrnn.generate_samples(
                max_gen_length=self.max_gen_length)


class save_model_weights(Callback):
    def __init__(self, weights_name):
        self.weights_name = weights_name

    def on_epoch_end(self, epoch, logs={}):
        if model_input_count(self.model) > 1:
            self.model = Model(inputs=self.model.input[0],
                               outputs=self.model.output[1])
        self.model.save_weights("{}_weights.hdf5".format(self.weights_name))
