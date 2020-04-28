import json
import re

import numpy as np
import tensorflow as tf
import tqdm
from pkg_resources import resource_filename
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelBinarizer
from tensorflow import config as config
from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence

from .model import textgenrnn_model
from .model_training import generate_sequences_from_texts
from .utils import (
    generate_after_epoch,
    save_model_weights,
    textgenrnn_encode_sequence,
    textgenrnn_generate,
    textgenrnn_texts_from_file,
    textgenrnn_texts_from_file_context,
)


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
                 name="textgenrnn",
                 allow_growth=None):

        if weights_path is None:
            weights_path = resource_filename(__name__,
                                             'textgenrnn_weights.hdf5')

        if vocab_path is None:
            vocab_path = resource_filename(__name__,
                                           'textgenrnn_vocab.json')

        if allow_growth is not None:
            c = tf.compat.v1.ConfigProto()
            c.gpu_options.allow_growth = True
            set_session(tf.compat.v1.Session(config=c))

        if config_path is not None:
            with open(config_path, 'r',
                      encoding='utf8', errors='ignore') as json_file:
                self.config = json.load(json_file)

        self.config.update({'name': name})
        self.default_config.update({'name': name})

        with open(vocab_path, 'r',
                  encoding='utf8', errors='ignore') as json_file:
            self.vocab = json.load(json_file)

        self.tokenizer = Tokenizer(filters='', lower=False, char_level=True)
        self.tokenizer.word_index = self.vocab
        self.num_classes = len(self.vocab) + 1
        self.model = textgenrnn_model(self.num_classes,
                                      cfg=self.config,
                                      weights_path=weights_path)
        self.indices_char = dict((self.vocab[c], c) for c in self.vocab)

    def generate(self, n=1, return_as_list=False, prefix=None,
                 temperature=[1.0, 0.5, 0.2, 0.2],
                 max_gen_length=300, interactive=False,
                 top_n=3, progress=True):
        gen_texts = []
        iterable = tqdm.trange(n) if progress and n > 1 else range(n)
        for _ in iterable:
            gen_text, _ = textgenrnn_generate(self.model,
                                              self.vocab,
                                              self.indices_char,
                                              temperature,
                                              self.config['max_length'],
                                              self.META_TOKEN,
                                              self.config['word_level'],
                                              self.config.get(
                                                  'single_text', False),
                                              max_gen_length,
                                              interactive,
                                              top_n,
                                              prefix)
            if not return_as_list:
                print("{}\n".format(gen_text))
            gen_texts.append(gen_text)
        if return_as_list:
            return gen_texts

    def generate_samples(self, n=3, temperatures=[0.2, 0.5, 1.0], **kwargs):
        for temperature in temperatures:
            print('#'*20 + '\nTemperature: {}\n'.format(temperature) +
                  '#'*20)
            self.generate(n, temperature=temperature, progress=False, **kwargs)

    def train_on_texts(self, texts, context_labels=None,
                       batch_size=128,
                       num_epochs=50,
                       verbose=1,
                       new_model=False,
                       gen_epochs=1,
                       train_size=1.0,
                       max_gen_length=300,
                       validation=True,
                       dropout=0.0,
                       via_new_model=False,
                       save_epochs=0,
                       multi_gpu=False,
                       **kwargs):

        if new_model and not via_new_model:
            self.train_new_model(texts,
                                 context_labels=context_labels,
                                 num_epochs=num_epochs,
                                 gen_epochs=gen_epochs,
                                 train_size=train_size,
                                 batch_size=batch_size,
                                 dropout=dropout,
                                 validation=validation,
                                 save_epochs=save_epochs,
                                 multi_gpu=multi_gpu,
                                 **kwargs)
            return

        if context_labels:
            context_labels = LabelBinarizer().fit_transform(context_labels)

        if self.config['word_level']:
            # If training word level, must add spaces around each
            # punctuation. https://stackoverflow.com/a/3645946/9314418
            punct = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\\n\\t\'‘’“”’–—…'
            for i in range(len(texts)):
                texts[i] = re.sub('([{}])'.format(punct), r' \1 ', texts[i])
                texts[i] = re.sub(' {2,}', ' ', texts[i])
            texts = [text_to_word_sequence(text, filters='') for text in texts]

        # calculate all combinations of text indices + token indices
        indices_list = [np.meshgrid(np.array(i), np.arange(
            len(text) + 1)) for i, text in enumerate(texts)]
        # indices_list = np.block(indices_list) # this hangs when indices_list is large enough
        # FIX BEGIN ------
        indices_list_o = np.block(indices_list[0])
        for i in range(len(indices_list)-1):
            tmp = np.block(indices_list[i+1])
            indices_list_o = np.concatenate([indices_list_o, tmp])
        indices_list = indices_list_o
        # FIX END ------

        # If a single text, there will be 2 extra indices, so remove them
        # Also remove first sequences which use padding
        if self.config['single_text']:
            indices_list = indices_list[self.config['max_length']:-2, :]

        indices_mask = np.random.rand(indices_list.shape[0]) < train_size

        if multi_gpu:
            num_gpus = len(config.get_visible_devices('GPU'))
            batch_size = batch_size * num_gpus

        gen_val = None
        val_steps = None
        if train_size < 1.0 and validation:
            indices_list_val = indices_list[~indices_mask, :]
            gen_val = generate_sequences_from_texts(
                texts, indices_list_val, self, context_labels, batch_size)
            val_steps = max(
                int(np.floor(indices_list_val.shape[0] / batch_size)), 1)

        indices_list = indices_list[indices_mask, :]

        num_tokens = indices_list.shape[0]
        assert num_tokens >= batch_size, "Fewer tokens than batch_size."

        level = 'word' if self.config['word_level'] else 'character'
        print("Training on {:,} {} sequences.".format(num_tokens, level))

        steps_per_epoch = max(int(np.floor(num_tokens / batch_size)), 1)

        gen = generate_sequences_from_texts(
            texts, indices_list, self, context_labels, batch_size)

        base_lr = 4e-3

        # scheduler function must be defined inline.
        def lr_linear_decay(epoch):
            return (base_lr * (1 - (epoch / num_epochs)))

        '''
        FIXME
        This part is a bit messy as we need to initialize the model within
        strategy.scope() when using multi-GPU. Can probably be cleaned up a bit.
        '''

        if context_labels is not None:
            if new_model:
                weights_path = None
            else:
                weights_path = "{}_weights.hdf5".format(self.config['name'])
                self.save(weights_path)


            if multi_gpu:
                from tensorflow import distribute as distribute
                strategy = distribute.MirroredStrategy()
                with strategy.scope():
                    parallel_model = textgenrnn_model(self.num_classes,
                                                      dropout=dropout,
                                                      cfg=self.config,
                                                      context_size=context_labels.shape[1],
                                                      weights_path=weights_path)
                    parallel_model.compile(loss='categorical_crossentropy',
                                           optimizer=Adam(lr=4e-3))
                model_t = parallel_model
                print("Training on {} GPUs.".format(num_gpus))
            else:
                model_t = self.model
        else:
            if multi_gpu:
                from tensorflow import distribute as distribute
                if new_model:
                    weights_path = None
                else:
                    weights_path = "{}_weights.hdf5".format(self.config['name'])

                strategy = distribute.MirroredStrategy()
                with strategy.scope():
                # Do not locate model/merge on CPU since sample sizes are small.
                    parallel_model = textgenrnn_model(self.num_classes,
                                                      cfg=self.config,
                                                      weights_path=weights_path)
                    parallel_model.compile(loss='categorical_crossentropy',
                                           optimizer=Adam(lr=4e-3))

                model_t = parallel_model
                print("Training on {} GPUs.".format(num_gpus))
            else:
                model_t = self.model

        model_t.fit(gen, steps_per_epoch=steps_per_epoch,
                              epochs=num_epochs,
                              callbacks=[
                                  LearningRateScheduler(
                                      lr_linear_decay),
                                  generate_after_epoch(
                                      self, gen_epochs,
                                      max_gen_length),
                                  save_model_weights(
                                      self, num_epochs,
                                      save_epochs)],
                              verbose=verbose,
                              max_queue_size=10,
                              validation_data=gen_val,
                              validation_steps=val_steps
                              )

        # Keep the text-only version of the model if using context labels
        if context_labels is not None:
            self.model = Model(inputs=self.model.input[0],
                               outputs=self.model.output[1])

    def train_new_model(self, texts, context_labels=None, num_epochs=50,
                        gen_epochs=1, batch_size=128, dropout=0.0,
                        train_size=1.0,
                        validation=True, save_epochs=0,
                        multi_gpu=False, **kwargs):
        self.config = self.default_config.copy()
        self.config.update(**kwargs)

        print("Training new model w/ {}-layer, {}-cell {}LSTMs".format(
            self.config['rnn_layers'], self.config['rnn_size'],
            'Bidirectional ' if self.config['rnn_bidirectional'] else ''
        ))

        # Create text vocabulary for new texts
        # if word-level, lowercase; if char-level, uppercase
        self.tokenizer = Tokenizer(filters='',
                                   lower=self.config['word_level'],
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
                                      dropout=dropout,
                                      cfg=self.config)

        # Save the files needed to recreate the model
        with open('{}_vocab.json'.format(self.config['name']),
                  'w', encoding='utf8') as outfile:
            json.dump(self.tokenizer.word_index, outfile, ensure_ascii=False)

        with open('{}_config.json'.format(self.config['name']),
                  'w', encoding='utf8') as outfile:
            json.dump(self.config, outfile, ensure_ascii=False)

        self.train_on_texts(texts, new_model=True,
                            via_new_model=True,
                            context_labels=context_labels,
                            num_epochs=num_epochs,
                            gen_epochs=gen_epochs,
                            train_size=train_size,
                            batch_size=batch_size,
                            dropout=dropout,
                            validation=validation,
                            save_epochs=save_epochs,
                            multi_gpu=multi_gpu,
                            **kwargs)

    def save(self, weights_path="textgenrnn_weights_saved.hdf5"):
        self.model.save_weights(weights_path)

    def load(self, weights_path):
        self.model = textgenrnn_model(self.num_classes,
                                      cfg=self.config,
                                      weights_path=weights_path)

    def reset(self):
        self.config = self.default_config.copy()
        self.__init__(name=self.config['name'])

    def train_from_file(self, file_path, header=True, delim="\n",
                        new_model=False, context=None,
                        is_csv=False, **kwargs):

        context_labels = None
        if context:
            texts, context_labels = textgenrnn_texts_from_file_context(
                file_path)
        else:
            texts = textgenrnn_texts_from_file(file_path, header,
                                               delim, is_csv)

        print("{:,} texts collected.".format(len(texts)))
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
        with open(destination_path, 'w', encoding="utf-8") as f:
            for text in texts:
                f.write("{}\n".format(text))

    def encode_text_vectors(self, texts, pca_dims=50, tsne_dims=None,
                            tsne_seed=None, return_pca=False,
                            return_tsne=False):

        # if a single text, force it into a list:
        if isinstance(texts, str):
            texts = [texts]

        vector_output = Model(inputs=self.model.input,
                              outputs=self.model.get_layer('attention').output)
        encoded_vectors = []
        maxlen = self.config['max_length']
        for text in texts:
            if self.config['word_level']:
                text = text_to_word_sequence(text, filters='')
            text_aug = [self.META_TOKEN] + list(text[0:maxlen])
            encoded_text = textgenrnn_encode_sequence(text_aug, self.vocab,
                                                      maxlen)
            encoded_vector = vector_output.predict(encoded_text)
            encoded_vectors.append(encoded_vector)

        encoded_vectors = np.squeeze(np.array(encoded_vectors), axis=1)
        if pca_dims is not None:
            assert len(texts) > 1, "Must use more than 1 text for PCA"
            pca = PCA(pca_dims)
            encoded_vectors = pca.fit_transform(encoded_vectors)

        if tsne_dims is not None:
            tsne = TSNE(tsne_dims, random_state=tsne_seed)
            encoded_vectors = tsne.fit_transform(encoded_vectors)

        return_objects = encoded_vectors
        if return_pca or return_tsne:
            return_objects = [return_objects]
        if return_pca:
            return_objects.append(pca)
        if return_tsne:
            return_objects.append(tsne)

        return return_objects

    def similarity(self, text, texts, use_pca=True):
        text_encoded = self.encode_text_vectors(text, pca_dims=None)
        if use_pca:
            texts_encoded, pca = self.encode_text_vectors(texts,
                                                          return_pca=True)
            text_encoded = pca.transform(text_encoded)
        else:
            texts_encoded = self.encode_text_vectors(texts, pca_dims=None)

        cos_similairity = cosine_similarity(text_encoded, texts_encoded)[0]
        text_sim_pairs = list(zip(texts, cos_similairity))
        text_sim_pairs = sorted(text_sim_pairs, key=lambda x: -x[1])
        return text_sim_pairs
