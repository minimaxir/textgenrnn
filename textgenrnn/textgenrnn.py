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
from .model_training import *
from .utils import *
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
        'word_level': False,
        'single_text': False
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
                                           self.config.get(
                                               'single_text', False),
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
                       max_gen_length=300,
                       **kwargs):

        if context_labels:
            context_labels = LabelBinarizer().fit_transform(context_labels)

        if self.config['word_level']:
            texts = [text_to_word_sequence(text, filters='') for text in texts]

        # calculate all combinations of text indices + token indices
        indices_list = [np.meshgrid(np.array(i), np.arange(
            len(text) + 1)) for i, text in enumerate(texts)]
        indices_list = np.block(indices_list)

        # If a single text, there will be 2 extra indices, so remove them
        # Also remove first sequences which use padding
        if self.config['single_text']:
            indices_list = indices_list[self.config['max_length']:-2, :]

        indices_list = indices_list[np.random.rand(
            indices_list.shape[0]) < prop_keep, :]
        num_tokens = indices_list.shape[0]

        level = 'word' if self.config['word_level'] else 'character'
        print("Training on {} {} sequences.".format(num_tokens, level))

        steps_per_epoch = max(int(np.floor(num_tokens / batch_size)), 1)

        gen = generate_sequences_from_texts(
            texts, indices_list, self, context_labels, batch_size)

        base_lr = 4e-3

        # scheduler function must be defined inline.
        def lr_linear_decay(epoch):
            return (base_lr * (1 - (epoch / num_epochs)))

        if context_labels:
            weights_path = resource_filename(__name__,
                                             'textgenrnn_weights.hdf5')
            if new_model:
                weights_path = None

            self.model = textgenrnn_model(self.num_classes,
                                          cfg=self.config,
                                          context_size=context_labels.shape[1],
                                          weights_path=weights_path)

        self.model.fit_generator(gen, steps_per_epoch=steps_per_epoch,
                                 epochs=num_epochs,
                                 callbacks=[
                                     LearningRateScheduler(
                                         lr_linear_decay),
                                     generate_after_epoch(
                                         self, gen_epochs,
                                         max_gen_length),
                                     save_model_weights(
                                         self.config['name'])],
                                 verbose=verbose,
                                 max_queue_size=2
                                 )

        # Keep the text-only version of the model if using context labels
        if context_labels:
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
            punct = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\\n\\t\'‘’“”’–—'
            for i in range(len(texts)):
                texts[i] = re.sub('([{}])'.format(punct), r' \1 ', texts[i])
                texts[i] = re.sub(' {2,}', ' ', texts[i])

        # Create text vocabulary for new texts
        self.tokenizer = Tokenizer(filters='',
                                   char_level=(not self.config['word_level']))
        self.tokenizer.fit_on_texts(texts)

        # Limit vocab to max_words
        max_words = self.config['max_words']
        self.tokenizer.word_index = {k: v for (
            k, v) in self.tokenizer.word_index.items() if v <= max_words}

        if not self.config.get('single_text', False):
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
                texts, single_text=True, **kwargs)
        else:
            self.train_on_texts(texts, single_text=True, **kwargs)

    def generate_to_file(self, destination_path, **kwargs):
        texts = self.generate(return_as_list=True, **kwargs)
        with open(destination_path, 'w') as f:
            for text in texts:
                f.write("{}\n".format(text))
