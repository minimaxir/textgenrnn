from keras.optimizers import RMSprop
from keras.layers import Input, Embedding, Dense, LSTM, CuDNNLSTM, concatenate
from keras.models import Model
from .AttentionWeightedAverage import AttentionWeightedAverage


def textgenrnn_model(weights_path, num_classes, maxlen=40,
                     optimizer=RMSprop(lr=4e-3, rho=0.99)):
    '''
    Builds the model architecture for textgenrnn and
    loads the specified weights for the model.
    '''

    input = Input(shape=(maxlen,), name='input')
    embedded = Embedding(num_classes, 100, input_length=maxlen,
                         trainable=True, name='embedding')(input)

    '''
    The normal LSTMs use sigmoid activations for parity with CuDNNLSTM:
    https://github.com/keras-team/keras/issues/8860
    https://github.com/keras-team/keras/pull/9112
    '''

    rnn_1 = LSTM(128, return_sequences=True,
                 recurrent_activation='sigmoid', name='rnn_1')(embedded)
    rnn_2 = LSTM(128, return_sequences=True,
                 recurrent_activation='sigmoid', name='rnn_2')(rnn_1)
    seq_concat = concatenate([embedded, rnn_1, rnn_2], name='rnn_concat')
    attention = AttentionWeightedAverage(name='attention')(seq_concat)

    output = Dense(num_classes, name='output', activation='softmax')(attention)

    model = Model(inputs=[input], outputs=[output])
    model.load_weights(weights_path, by_name=True)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model
