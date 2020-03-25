import numpy as np
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer

from .utils import textgenrnn_encode_cat


def generate_sequences_from_texts(texts, indices_list,
                                  textgenrnn, context_labels,
                                  batch_size=128):
    is_words = textgenrnn.config['word_level']
    is_single = textgenrnn.config['single_text']
    max_length = textgenrnn.config['max_length']
    meta_token = textgenrnn.META_TOKEN

    if is_words:
        new_tokenizer = Tokenizer(filters='', char_level=True)
        new_tokenizer.word_index = textgenrnn.vocab
    else:
        new_tokenizer = textgenrnn.tokenizer

    while True:
        np.random.shuffle(indices_list)

        X_batch = []
        Y_batch = []
        context_batch = []
        count_batch = 0

        for row in range(indices_list.shape[0]):
            text_index = indices_list[row, 0]
            end_index = indices_list[row, 1]

            text = texts[text_index]

            if not is_single:
                text = [meta_token] + list(text) + [meta_token]

            if end_index > max_length:
                x = text[end_index - max_length: end_index + 1]
            else:
                x = text[0: end_index + 1]
            y = text[end_index + 1]

            if y in textgenrnn.vocab:
                x = process_sequence([x], textgenrnn, new_tokenizer)
                y = textgenrnn_encode_cat([y], textgenrnn.vocab)

                X_batch.append(x)
                Y_batch.append(y)

                if context_labels is not None:
                    context_batch.append(context_labels[text_index])

                count_batch += 1

                if count_batch % batch_size == 0:
                    X_batch = np.squeeze(np.array(X_batch))
                    Y_batch = np.squeeze(np.array(Y_batch))
                    context_batch = np.squeeze(np.array(context_batch))

                    # print(X_batch.shape)

                    if context_labels is not None:
                        yield ([X_batch, context_batch], [Y_batch, Y_batch])
                    else:
                        yield (X_batch, Y_batch)
                    X_batch = []
                    Y_batch = []
                    context_batch = []
                    count_batch = 0


def process_sequence(X, textgenrnn, new_tokenizer):
    X = new_tokenizer.texts_to_sequences(X)
    X = sequence.pad_sequences(
        X, maxlen=textgenrnn.config['max_length'])

    return X
