import csv
import re
from random import shuffle

import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import sequence
from tqdm import trange


def textgenrnn_sample(preds, temperature, interactive=False, top_n=3):
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

    if not interactive:
        index = np.argmax(probas)

        # prevent function from being able to choose 0 (placeholder)
        # choose 2nd best index from preds
        if index == 0:
            index = np.argsort(preds)[-2]
    else:
        # return list of top N chars/words
        # descending order, based on probability
        index = (-preds).argsort()[:top_n]

    return index


def textgenrnn_generate(model, vocab,
                        indices_char, temperature=0.5,
                        maxlen=40, meta_token='<s>',
                        word_level=False,
                        single_text=False,
                        max_gen_length=300,
                        interactive=False,
                        top_n=3,
                        prefix=None,
                        synthesize=False,
                        stop_tokens=[' ', '\n']):
    '''
    Generates and returns a single text.
    '''

    collapse_char = ' ' if word_level else ''
    end = False

    # If generating word level, must add spaces around each punctuation.
    # https://stackoverflow.com/a/3645946/9314418
    if word_level and prefix:
        punct = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\\n\\t\'‘’“”’–—'
        prefix = re.sub('([{}])'.format(punct), r' \1 ', prefix)
        prefix = re.sub(' {2,}', r' ', prefix)
        prefix_t = [x.lower() for x in prefix.split(' ')]

    if not word_level and prefix:
        prefix_t = list(prefix)

    if single_text:
        text = prefix_t if prefix else ['']
        max_gen_length += maxlen
    else:
        text = [meta_token] + prefix_t if prefix else [meta_token]

    if not isinstance(temperature, list):
        temperature = [temperature]

    if len(model.inputs) > 1:
        model = Model(inputs=model.inputs[0], outputs=model.outputs[1])

    while not end and len(text) < max_gen_length:
        encoded_text = textgenrnn_encode_sequence(text[-maxlen:],
                                                  vocab, maxlen)
        next_temperature = temperature[(len(text) - 1) % len(temperature)]

        if not interactive:
            # auto-generate text without user intervention
            next_index = textgenrnn_sample(
                model.predict(encoded_text, batch_size=1)[0],
                next_temperature)
            next_char = indices_char[next_index]
            text += [next_char]
            if next_char == meta_token or len(text) >= max_gen_length:
                end = True
            gen_break = (next_char in stop_tokens or word_level or
                         len(stop_tokens) == 0)
            if synthesize and gen_break:
                break
        else:
            # ask user what the next char/word should be
            options_index = textgenrnn_sample(
                model.predict(encoded_text, batch_size=1)[0],
                next_temperature,
                interactive=interactive,
                top_n=top_n
            )
            options = [indices_char[idx] for idx in options_index]
            print('Controls:\n\ts: stop.\tx: backspace.\to: write your own.')
            print('\nOptions:')

            for i, option in enumerate(options, 1):
                print('\t{}: {}'.format(i, option))

            print('\nProgress: {}'.format(collapse_char.join(text)[3:]))
            print('\nYour choice?')
            user_input = input('> ')

            try:
                user_input = int(user_input)
                next_char = options[user_input-1]
                text += [next_char]
            except ValueError:
                if user_input == 's':
                    next_char = '<s>'
                    text += [next_char]
                elif user_input == 'o':
                    other = input('> ')
                    text += [other]
                elif user_input == 'x':
                    try:
                        del text[-1]
                    except IndexError:
                        pass
                else:
                    print('That\'s not an option!')

    # if single text, ignore sequences generated w/ padding
    # if not single text, remove the <s> meta_tokens
    if single_text:
        text = text[maxlen:]
    else:
        text = text[1:]
        if meta_token in text:
            text.remove(meta_token)

    text_joined = collapse_char.join(text)

    # If word level, remove spaces around punctuation for cleanliness.
    if word_level:
        left_punct = "!%),.:;?@\]_}\\n\\t'"
        right_punct = "$(\[_\\n\\t'"
        punct = '\\n\\t'

        text_joined = re.sub(" ([{}]) ".format(
            punct), r'\1', text_joined)
        text_joined = re.sub(" ([{}])".format(
            left_punct), r'\1', text_joined)
        text_joined = re.sub("([{}]) ".format(
            right_punct), r'\1', text_joined)
        text_joined = re.sub('" (.+?) "', 
            r'"\1"', text_joined)

    return text_joined, end


def textgenrnn_encode_sequence(text, vocab, maxlen):
    '''
    Encodes a text into the corresponding encoding for prediction with
    the model.
    '''

    encoded = np.array([vocab.get(x, 0) for x in text])
    return sequence.pad_sequences([encoded], maxlen=maxlen)


def textgenrnn_texts_from_file(file_path, header=True,
                               delim='\n', is_csv=False):
    '''
    Retrieves texts from a newline-delimited file and returns as a list.
    '''

    with open(file_path, 'r', encoding='utf8', errors='ignore') as f:
        if header:
            f.readline()
        if is_csv:
            texts = []
            reader = csv.reader(f)
            for row in reader:
                if row:
                    texts.append(row[0])
        else:
            text_data = f.read()
            texts = text_data.split(delim)

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
            if row:
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


def synthesize(textgens, n=1, return_as_list=False, prefix='',
               temperature=[0.5, 0.2, 0.2], max_gen_length=300,
               progress=True, stop_tokens=[' ', '\n']):
    """Synthesizes texts using an ensemble of input models.
    """

    gen_texts = []
    iterable = trange(n) if progress and n > 1 else range(n)
    for _ in iterable:
        shuffle(textgens)
        gen_text = prefix
        end = False
        textgen_i = 0
        while not end:
            textgen = textgens[textgen_i % len(textgens)]
            gen_text, end = textgenrnn_generate(textgen.model,
                                                textgen.vocab,
                                                textgen.indices_char,
                                                temperature,
                                                textgen.config['max_length'],
                                                textgen.META_TOKEN,
                                                textgen.config['word_level'],
                                                textgen.config.get(
                                                    'single_text', False),
                                                max_gen_length,
                                                prefix=gen_text,
                                                synthesize=True,
                                                stop_tokens=stop_tokens)
            textgen_i += 1
        if not return_as_list:
            print("{}\n".format(gen_text))
        gen_texts.append(gen_text)
    if return_as_list:
        return gen_texts


def synthesize_to_file(textgens, destination_path, **kwargs):
    texts = synthesize(textgens, return_as_list=True, **kwargs)
    with open(destination_path, 'w') as f:
        for text in texts:
            f.write("{}\n".format(text))


class generate_after_epoch(Callback):
    def __init__(self, textgenrnn, gen_epochs, max_gen_length):
        super().__init__()
        self.textgenrnn = textgenrnn
        self.gen_epochs = gen_epochs
        self.max_gen_length = max_gen_length

    def on_epoch_end(self, epoch, logs={}):
        if self.gen_epochs > 0 and (epoch+1) % self.gen_epochs == 0:
            self.textgenrnn.generate_samples(
                max_gen_length=self.max_gen_length)


class save_model_weights(Callback):
    def __init__(self, textgenrnn, num_epochs, save_epochs):
        super().__init__()
        self.textgenrnn = textgenrnn
        self.weights_name = textgenrnn.config['name']
        self.num_epochs = num_epochs
        self.save_epochs = save_epochs

    def on_epoch_end(self, epoch, logs={}):
        if len(self.textgenrnn.model.inputs) > 1:
            self.textgenrnn.model = Model(inputs=self.model.input[0],
                                          outputs=self.model.output[1])
        if self.save_epochs > 0 and (epoch+1) % self.save_epochs == 0 and self.num_epochs != (epoch+1):
            print("Saving Model Weights — Epoch #{}".format(epoch+1))
            self.textgenrnn.model.save_weights(
                "{}_weights_epoch_{}.hdf5".format(self.weights_name, epoch+1))
        else:
            self.textgenrnn.model.save_weights(
                "{}_weights.hdf5".format(self.weights_name))
