import numpy as np
import csv
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
    MAX_LENGTH = 300
    META_TOKEN = '<s>'

    def __init__(self, file_path, vocab_path):
        self.file_path = file_path
