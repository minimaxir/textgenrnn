from keras.layers import Input, Embedding, LSTM, Dense, GRU
from keras.optimizers import Nadam, Adam
from keras.callbacks import ModelCheckpoint, Callback
from keras.callbacks import LearningRateScheduler, LambdaCallback
from keras.models import Model, load_model
from keras.preprocessing import sequence
from random import seed, random, sample
import numpy as np
import json
import h5py
import csv


class textgenrnn():
    MAX_LENGTH = 150
    META_TOKEN = '<s>'
    BATCH_SIZE = 128
    EPOCHS = 5

    def __init__(self, model_path, vocab_path):
        self.model = load_model(model_path)
        self.tokenizer = Tokenizer()
        self.tokenizer.word_index = json.loads(vocab_path)
        self.num_classes = len(self.tokenizer.word_index) - 1

    def generate(self, text, prefix=None, temperature=1.0, n=1,
                 return_as_list=False):
        gen_texts = []
        for i in range(n - 1):
            gen_text = [META_TOKEN] + list(prefix) if prefix else [META_TOKEN]
            next_char = ''
            while next_char is not META_TOKEN:
                encoded_text = textgenrnn_encode_sequence(gen_text,
                                                          self.tokenizer.word_index)
                probs = self.model.predict(encoded_text)[0]
                next_char_index = textgenrnn_sample(probs)
                next_char = indices_char[next_char_index]
                gen_text += [next_char]
            print(gen_text[1:-1])
            gen_texts.append(gen_text[1:-1])
        return gen_texts if return_as_list

    def train(self, texts, fc_layers=[256, 128]):

        # Encode texts as valid X and y.
        X = []
        y = []

        for text in texts:
            subset_x, subset_y = textgenrnn_encode_training(text, meta_token)
            for i in range(len(subset_x)):
                X.append(subset_x[i])
                y.append(subset_y[i])

        X = np.array(X)
        y = np.array(y)

        X = self.tokenizer.texts_to_sequences(X)
        X = sequence.pad_sequences(X, maxlen=MAX_LENGTH + 1)
        y = np.array([self.tokenizer.word_index[x] for x in y])

        # Append Dense layers to model and retrain.
        self.model.layers.pop()
        output = self.model.layers[-1].output

        for i, layer_dims in enumerate(fc_layers):
            output = Dense(layer_dims, use_bias=False,
                           name='hidden_' + (i + 1))(output)
            output = BatchNormalization()(output)
            output = Activation('relu')(output)

        output = Dense(num_classes,
                       activation='softmax', name='output')(output)

        self.model = Model(inputs=self.model.input, outputs=output)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=Adam(lr=0.01, decay=1e-6))

        self.model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS)

    def train_from_file(self, file_path, **kwargs):
        files = [file_path] if file_path is str else file_path
        texts = []
        for file in files:
            texts += textgenrnn_texts_from_file(file, **kwargs)
        self.train(texts, **kwargs)

    def generate_to_file(self, destination_path, **kwargs):
        texts = self.generate(**kwargs, return_as_list=True)
        with open(destination_path, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write("{}\n".format(text))


def textgenrnn_model(weights_path, num_classes, maxlen=40):
    '''
    Builds the model architecture for textgenrnn and
    loads the pretrained weights.
    '''

    input = Input(shape=(maxlen,), name='input')
    embedded = Embedding(num_classes, 100, name='embedding',
                         input_length=maxlen)(input)

    rnn = GRU(128, return_sequences=False, name='rnn')(embedded)

    output = Dense(num_classes, name='output', activation='softmax')(rnn_1)

    model = Model(inputs=[input], outputs=[output])
    optimizer = Nadam()

    model.load_weights(weights_path, by_name=True)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer)


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

    with open(file_path, 'r', encoding="utf-8") as f:
        f.readline() if header
        texts = [line.rstrip('\n') for line in f]
    return texts
