import numpy as np
import csv
import json
from random import sample, seed
from keras.preprocessing import sequence
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.layers.wrappers import Bidirectional
from keras.models import load_model


class docrnn():
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
                encoded_text = docrnn_encode_sequence(gen_text,
                                                      self.tokenizer.word_index)
                next_char = indices_char[sample(model.predict(
                    encoded_text)[0])]
                gen_text += [next_char]
            print(gen_text[1:-1])
            gen_texts.append(gen_text[1:-1])
        return gen_texts if return_as_list

    def train(self, texts, fc_layers=[256, 128]):
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


def docrnn_sample(preds, temperature):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def docrnn_encode_sequence(text, vocab, meta_token='<s>', maxlen=150):
    encoded = [[vocab.get(x, 0) for x in ([meta_token] + list(text))]]
    return sequence.pad_sequences(encoded, maxlen=maxlen)
