"""
Defination of NN model
"""
from keras.layers import Dense, Dropout, Activation, Input, SimpleRNN
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential, Model
from keras import regularizers


def get_lstm(units):
    """LSTM(Long Short-Term Memory)
    Build LSTM Model.

    # Arguments
        units: List(int), number of input, output and hidden units.
    # Returns
        model: Model, nn model.
    """

    model = Sequential()
    model.add(LSTM(units[1], input_shape=(units[0], 1), return_sequences=True))
    model.add(LSTM(units[2]))
    model.add(Dropout(0.2))
    model.add(Dense(units[3], activation='sigmoid'))

    return model


def get_gru(units):
    """GRU(Gated Recurrent Unit)
    Build GRU Model.

    # Arguments
        units: List(int), number of input, output and hidden units.
    # Returns
        model: Model, nn model.
    """

    model = Sequential()
    model.add(GRU(units[1], input_shape=(units[0], 1), return_sequences=True))
    model.add(GRU(units[2]))
    model.add(Dropout(0.2))
    model.add(Dense(units[3], activation='sigmoid'))

    return model


def _get_sae(inputs, hidden, output):
    """SAE(Auto-Encoders)
    Build SAE Model.

    # Arguments
        inputs: Integer, number of input units.
        hidden: Integer, number of hidden units.
        output: Integer, number of output units.
    # Returns
        model: Model, nn model.
    """

    model = Sequential()
    model.add(Dense(hidden, input_dim=inputs, name='hidden'))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(output, activation='sigmoid'))

    return model


def get_saes(layers):
    """SAEs(Stacked Auto-Encoders)
    Build SAEs Model.

    # Arguments
        layers: List(int), number of input, output and hidden units.
    # Returns
        models: List(Model), List of SAE and SAEs.
    """
    sae1 = _get_sae(layers[0], layers[1], layers[-1])
    sae2 = _get_sae(layers[1], layers[2], layers[-1])
    sae3 = _get_sae(layers[2], layers[3], layers[-1])

    saes = Sequential()
    saes.add(Dense(layers[1], input_dim=layers[0], name='hidden1'))
    saes.add(Activation('sigmoid'))
    saes.add(Dense(layers[2], name='hidden2'))
    saes.add(Activation('sigmoid'))
    saes.add(Dense(layers[3], name='hidden3'))
    saes.add(Activation('sigmoid'))
    saes.add(Dropout(0.2))
    saes.add(Dense(layers[4], activation='sigmoid'))

    models = [sae1, sae2, sae3, saes]

    return models


def get_saes_fixed(input_dim, hidden_layers):
    """
    Build a stacked autoencoder (SAEs) model with encoder and decoder.

    # Arguments
    input_dim: int, number of input units.
    hidden_layers: List(int), number of hidden units.

    # Returns
    models: List(Model), List of SAEs and SAEs.The complete stacked autoencoder model.
    """

    # Input layer
    input_layer = Input(shape=(input_dim,), name='input')

    # Encoder
    encoder = input_layer
    for units in hidden_layers:
        encoder = Dense(units, activation='relu', activity_regularizer=regularizers.l1(1e-5))(encoder)

    # Bottleneck layer (Code layer)
    bottleneck = Dense(hidden_layers[-1], activation='relu', activity_regularizer=regularizers.l1(1e-5))(encoder)

    # Decoder (mirroring the encoder)
    decoder = bottleneck
    for units in reversed(hidden_layers[:-1]):
        decoder = Dense(units, activation='relu')(decoder)

    # Output layer (reconstruction of the input)
    output_layer = Dense(1, activation='sigmoid')(decoder)

    # Full autoencoder model
    autoencoder = Model(inputs=input_layer, outputs=output_layer)

    return autoencoder


def get_rnn(units):
    """RNN(Recurrent Neural Network)
    Build RNN Model.

    # Arguments
    units: List(int), number of input, output and hidden units.

    # Returns
    model: Model, RNN model."""

    model = Sequential()

    # First RNN layer (return sequences for stacked RNN)
    model.add(SimpleRNN(units[1], input_shape=(units[0], 1), return_sequences=True))

    # Second RNN layer
    model.add(SimpleRNN(units[2]))

    # Dropout for regularization
    model.add(Dropout(0.2))

    # Dense output layer with sigmoid activation
    model.add(Dense(units[3], activation='sigmoid'))

    return model
