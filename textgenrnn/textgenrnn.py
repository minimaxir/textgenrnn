from keras.callbacks import LearningRateScheduler, Callback
from keras.models import Model, load_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras import backend as K
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import json
import h5py
from pkg_resources import resource_filename
from .model import textgenrnn_model
import csv
import re


class textgenrnn:
    META_TOKEN = '<s>'
    config = {
        'rnn_layers': 2,
        'rnn_size': 128,
        'rnn_bidirectional': False,
        'max_length': 40,
        'max_words': 10000,
        'dim_embeddings': 100,
        'word_level': False
    }
    default_config = config.copy()

    def __init__(self, weights_path=None,
                 vocab_path=None,
                 config_path=None,
                 name="textgenrnn"):

        if weights_path is None:
            weights_path = resource_filename(__name__,
                                             'textgenrnn_weights.hdf5')

        if vocab_path is None:
            vocab_path = resource_filename(__name__,
                                           'textgenrnn_vocab.json')

        if config_path is not None:
            with open(config_path, 'r',
                      encoding='utf8', errors='ignore') as json_file:
                self.config = json.load(json_file)

        self.config.update({'name': name})
        self.default_config.update({'name': name})

        with open(vocab_path, 'r',
                  encoding='utf8', errors='ignore') as json_file:
            self.vocab = json.load(json_file)

        self.tokenizer = Tokenizer(filters='', char_level=True)
        self.tokenizer.word_index = self.vocab
        self.num_classes = len(self.vocab) + 1
        self.model = textgenrnn_model(self.num_classes,
                                      cfg=self.config,
                                      weights_path=weights_path)
        self.indices_char = dict((self.vocab[c], c) for c in self.vocab)

    def generate(self, n=1, return_as_list=False, prefix=None,
                 temperature=0.5, max_gen_length=300):
        gen_texts = []
        for _ in range(n):
            gen_text = textgenrnn_generate(self.model,
                                           self.vocab,
                                           self.indices_char,
                                           prefix,
                                           temperature,
                                           self.config['max_length'],
                                           self.META_TOKEN,
                                           self.config['word_level'],
                                           max_gen_length)
            if not return_as_list:
                print("{}\n".format(gen_text))
            gen_texts.append(gen_text)
        if return_as_list:
            return gen_texts

    def generate_samples(self, n=3, temperatures=[0.2, 0.5, 1.0], **kwargs):
        for temperature in temperatures:
            print('#'*20 + '\nTemperature: {}\n'.format(temperature) +
                  '#'*20)
            self.generate(n, temperature=temperature, **kwargs)

    def train_on_texts(self, texts, context_labels=None,
                       batch_size=128,
                       num_epochs=50,
                       verbose=1,
                       new_model=False,
                       gen_epochs=1,
                       prop_keep=1.0,
                       **kwargs):

        is_words = self.config['word_level']
        max_length = self.config['max_length']

        # Encode chars as X and y.
        X = []
        X_context = []
        y = []

        for i, text in enumerate(texts):
            subset_x, subset_y = textgenrnn_encode_training(text,
                                                            is_words,
                                                            self.META_TOKEN,
                                                            max_length)

            for j in range(len(subset_x)):
                if np.random.rand() < prop_keep:
                    X.append(subset_x[j])
                    y.append(subset_y[j])
                    if context_labels is not None:
                        X_context.append(context_labels[i])

        X = np.array(X)
        X_context = np.array(X_context)
        y = np.array(y)

        # Remake the tokenizer to avoid reprocessing word tokens
        if is_words:
            new_tokenizer = Tokenizer(filters='', char_level=True)
            new_tokenizer.word_index = self.vocab
        else:
            new_tokenizer = self.tokenizer

        X = new_tokenizer.texts_to_sequences(X)
        X = sequence.pad_sequences(X, maxlen=max_length)
        y = textgenrnn_encode_cat(y, self.vocab)

        if context_labels is not None:
            X_context_lb = LabelBinarizer().fit(context_labels)
            X_context = X_context_lb.transform(X_context)

        base_lr = 4e-3

        # scheduler function must be defined inline.
        def lr_linear_decay(epoch):
            return (base_lr * (1 - (epoch / num_epochs)))

        if context_labels is None:
            self.model.fit(X, y, batch_size=batch_size, epochs=num_epochs,
                           callbacks=[LearningRateScheduler(lr_linear_decay),
                                      generate_after_epoch(self, gen_epochs),
                                      save_model_weights(self.config['name'])],
                           verbose=verbose)
        else:
            weights_path = resource_filename(__name__,
                                             'textgenrnn_weights.hdf5')

            if new_model:
                weights_path = None

            self.model = textgenrnn_model(self.num_classes,
                                          cfg=self.config,
                                          context_size=X_context.shape[1],
                                          weights_path=weights_path)

            self.model.fit([X, X_context], [y, y],
                           batch_size=batch_size, epochs=num_epochs,
                           callbacks=[LearningRateScheduler(lr_linear_decay),
                                      generate_after_epoch(self, gen_epochs),
                                      save_model_weights(self.config['name'])],
                           verbose=verbose)

            # Keep the text-only version of the model
            self.model = Model(inputs=self.model.input[0],
                               outputs=self.model.output[1])

    def train_new_model(self, texts, context_labels=None, num_epochs=50,
                        gen_epochs=1, batch_size=128, **kwargs):
        self.config = self.default_config.copy()
        self.config.update(**kwargs)
        print("Training new model w/ {}-layer, {}-cell {}LSTMs".format(
            self.config['rnn_layers'], self.config['rnn_size'],
            'Bidirectional ' if self.config['rnn_bidirectional'] else ''
        ))

        # If training word level, must add spaces around each punctuation.
        # https://stackoverflow.com/a/3645946/9314418

        if self.config['word_level']:
            punct = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'
            for i in range(len(texts)):
                texts[i] = re.sub('([{}])'.format(punct), r' \1 ', texts[i])
                texts[i] = re.sub('\s{2,}', ' ', texts[i])

        # Create text vocabulary for new texts
        self.tokenizer = Tokenizer(filters='',
                                   char_level=(not self.config['word_level']))
        self.tokenizer.fit_on_texts(texts)
        self.tokenizer.word_index[self.META_TOKEN] = len(
            self.tokenizer.word_index) + 1
        self.vocab = self.tokenizer.word_index
        self.num_classes = len(self.vocab) + 1
        self.indices_char = dict((self.vocab[c], c) for c in self.vocab)

        # Create a new, blank model w/ given params
        self.model = textgenrnn_model(self.num_classes,
                                      cfg=self.config)

        # Save the files needed to recreate the model
        with open('{}_vocab.json'.format(self.config['name']),
                  'w') as outfile:
            json.dump(self.tokenizer.word_index, outfile, ensure_ascii=False)

        with open('{}_config.json'.format(self.config['name']),
                  'w') as outfile:
            json.dump(self.config, outfile, ensure_ascii=False)

        self.train_on_texts(texts, new_model=True,
                            context_labels=context_labels,
                            num_epochs=num_epochs,
                            gen_epochs=gen_epochs,
                            batch_size=batch_size,
                            **kwargs)

    def save(self, weights_path="textgenrnn_weights_saved.hdf5"):
        self.model.save_weights(weights_path)

    def load(self, weights_path):
        self.model = textgenrnn_model(weights_path, self.num_classes)

    def reset(self):
        self.config = self.default_config.copy()
        self.__init__(name=self.config['name'])

    def train_from_file(self, file_path, header=True, delim="\n",
                        new_model=False, context=None, **kwargs):

        context_labels = None
        if context:
            texts, context_labels = textgenrnn_texts_from_file_context(
                file_path)
        else:
            texts = textgenrnn_texts_from_file(file_path, header, delim)

        print("{} texts collected.".format(len(texts)))
        if new_model:
            self.train_new_model(
                texts, context_labels=context_labels, **kwargs)
        else:
            self.train_on_texts(texts, context_labels=context_labels, **kwargs)

    def train_from_largetext_file(self, file_path, new_model=True, **kwargs):
        with open(file_path, 'r', encoding='utf8', errors='ignore') as f:
            texts = [f.read()]

        if new_model:
            self.train_new_model(
                texts, **kwargs)
        else:
            self.train_on_texts(texts, **kwargs)

    def generate_to_file(self, destination_path, **kwargs):
        texts = self.generate(return_as_list=True, **kwargs)
        with open(destination_path, 'w') as f:
            for text in texts:
                f.write("{}\n".format(text))


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
    index = -1

    # prevent function from being able to choose 0 (placeholder)
    while index < 1:
        probas = np.random.multinomial(1, preds, 1)
        index = np.argmax(probas)
    return index


def textgenrnn_generate(model, vocab,
                        indices_char, prefix=None, temperature=0.5,
                        maxlen=40, meta_token='<s>',
                        word_level=False,
                        max_gen_length=300):
    '''
    Generates and returns a single text.
    '''

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

    return collapse_char.join(text[1:-1])


def textgenrnn_encode_sequence(text, vocab, maxlen):
    '''
    Encodes a text into the corresponding encoding for prediction with
    the model.
    '''

    encoded = np.array([vocab.get(x, 0) for x in text])
    return sequence.pad_sequences([encoded], maxlen=maxlen)


def textgenrnn_encode_training(text,
                               word_level=False,
                               meta_token='<s>', max_length=40):
    '''
    Encodes a list of texts into a list of texts, and the next character
    in those texts.
    '''

    if word_level:
        text_aug = text_to_word_sequence(text, filters='')
        text_aug = [meta_token] + text_aug + [meta_token]
    else:
        text_aug = [meta_token] + list(text) + [meta_token]
    chars = []
    next_char = []

    for i in range(len(text_aug) - 1):
        if i > max_length:
            chars.append(text_aug[i - max_length: i + 1])
        else:
            chars.append(text_aug[0:i + 1])
        next_char.append(text_aug[i + 1])

    return chars, next_char


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
    def __init__(self, textgenrnn, gen_epochs):
        self.textgenrnn = textgenrnn
        self.gen_epochs = gen_epochs

    def on_epoch_end(self, epoch, logs={}):
        if self.gen_epochs > 0 and (epoch+1) % self.gen_epochs == 0:
            self.textgenrnn.generate_samples()


class save_model_weights(Callback):
    def __init__(self, weights_name):
        self.weights_name = weights_name

    def on_epoch_end(self, epoch, logs={}):
        if model_input_count(self.model) > 1:
            self.model = Model(inputs=self.model.input[0],
                               outputs=self.model.output[1])
        self.model.save_weights("{}_weights.hdf5".format(self.weights_name))
