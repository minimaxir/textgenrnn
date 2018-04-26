from keras.callbacks import LearningRateScheduler, Callback
from keras.models import Model, load_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils import Sequence
from keras import backend as K
from .utils import textgenrnn_encode_training, textgenrnn_encode_cat
import numpy as np


def generate_sequences_from_texts(texts, textgenrnn, context_labels=None,
                                  batch_size=128):
    while True:
        idx = np.random.choice(range(len(texts)), len(texts),
                               replace=False)

        texts = np.array(texts)[idx]

        if context_labels is not None:
            context_labels = np.array(context_labels)[idx]

        is_words = textgenrnn.config['word_level']
        max_length = textgenrnn.config['max_length']
        meta_token = textgenrnn.META_TOKEN

        # Remake the tokenizer to avoid reprocessing word tokens
        if is_words:
            new_tokenizer = Tokenizer(filters='', char_level=True)
            new_tokenizer.word_index = textgenrnn.vocab
        else:
            new_tokenizer = textgenrnn.tokenizer

        X_batch = []
        Y_batch = []
        context_batch = []
        count_batch = 0

        for i, text in enumerate(texts):
            subset_x, subset_y = textgenrnn_encode_training(text,
                                                            is_words,
                                                            meta_token,
                                                            max_length)

            range_random = np.random.choice(range(len(subset_x)),
                                            len(subset_x),
                                            replace=False)

            for j in range_random:
                x = process_sequence([subset_x[j]],
                                     textgenrnn, new_tokenizer)

                y = textgenrnn_encode_cat([subset_y[j]], textgenrnn.vocab)

                X_batch.append(x)
                Y_batch.append(y)

                if context_labels is not None:
                    context_batch.append(context_labels[i])

                count_batch += 1

                if count_batch % batch_size == 0:
                    X_batch = np.squeeze(np.array(X_batch))
                    Y_batch = np.squeeze(np.array(Y_batch))
                    context_batch = np.squeeze(np.array(context_batch))

                    if context_labels is not None:
                        yield ([X_batch, context_batch], [Y_batch, Y_batch])
                    else:
                        yield (X_batch, Y_batch)
                    X_batch = []
                    Y_batch = []
                    context_batch = []
                    count_batch = 0

        #  run if number of characters in dataset < batch_size
        X_batch = np.squeeze(np.array(X_batch))
        Y_batch = np.squeeze(np.array(Y_batch))
        context_batch = np.squeeze(np.array(context_batch))

        if context_labels is not None:
            yield ([X_batch, context_batch], [Y_batch, Y_batch])
        else:
            yield (X_batch, Y_batch)


def process_sequence(X, textgenrnn, new_tokenizer):
    X = np.array(X)
    X = new_tokenizer.texts_to_sequences(X)
    X = sequence.pad_sequences(
        X, maxlen=textgenrnn.config['max_length'])

    return X
