"""
Train the NN model.
"""
import sys
import warnings
import argparse
import numpy as np
import pandas as pd
from data.data import process_data
from model import model
from keras.models import Model
import os
from keras.callbacks import EarlyStopping
warnings.filterwarnings("ignore")


def get_scats_sites(data_dir):
    """Get SCATS sites based on file names from the directory"""
    # Get all the train file names from the folder
    files = os.listdir(data_dir)
    train_files = [f for f in files if 'train' in f]

    # Extract SCATS IDs (e.g., '2000', '2200', etc.)
    scats_sites = [f.split('_')[0] for f in train_files]

    return scats_sites

def train_model(model, X_train, y_train, name, config, site):
    """train
    train a single model.

    # Arguments
        model: Model, NN model to train.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
    """

    model.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
    # early = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')
    hist = model.fit(
        X_train, y_train,
        batch_size=config["batch"],
        epochs=config["epochs"],
        validation_split=0.05)

    model.save(os.path.join(os.path.dirname(__file__), 'model', 'sites_models', f'{name}_{site}.h5'))
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv(os.path.join(os.path.dirname(__file__), 'model', 'sites_models', f'{name}_{site}_loss.csv'), encoding='utf-8', index=False)


def train_seas(models, X_train, y_train, name, config, site):
    """train
    train the SAEs model.

    # Arguments
        models: List, list of SAE model.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
    """

    temp = X_train
    # early = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')

    for i in range(len(models) - 1):
        if i > 0:
            p = models[i - 1]
            hidden_layer_model = Model(input=p.input,
                                       output=p.get_layer('hidden').output)
            temp = hidden_layer_model.predict(temp)

        m = models[i]
        m.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])

        m.fit(temp, y_train, batch_size=config["batch"],
              epochs=config["epochs"],
              validation_split=0.05)

        models[i] = m

    saes = models[-1]
    for i in range(len(models) - 1):
        weights = models[i].get_layer('hidden').get_weights()
        saes.get_layer('hidden%d' % (i + 1)).set_weights(weights)

    train_model(saes, X_train, y_train, name, config, site)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="saes",
        help="Model to train.")
    args = parser.parse_args()

    lag = 12
    config = {"batch": 128, "epochs": 10}

    # Get all SCATS sites (by extracting unique IDs from file names)
    data_dir = 'data/splitted_data'
    scats_sites = get_scats_sites(data_dir)

    # Loop through each SCATS site and train the model
    for site in scats_sites:
        train_file = os.path.join(data_dir, f'{site}_train.csv')
        test_file = os.path.join(data_dir, f'{site}_test.csv')

        # Process data for each SCATS site
        X_train, y_train, _, _, _ = process_data(train_file, test_file, lag)

        # Reshape input data based on the model type
        if args.model == 'lstm':
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            m = model.get_lstm([lag, 64, 64, 1])
            train_model(m, X_train, y_train, args.model, config, site)

        elif args.model == 'gru':
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            m = model.get_gru([lag, 64, 64, 1])
            train_model(m, X_train, y_train, args.model, config, site)

        elif args.model == 'saes':
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
            m = model.get_saes([lag, 400, 400, 400, 1])
            train_seas(m, X_train, y_train, args.model, config, site)

        print(f"Finished training model for SCATS site: {site}")


if __name__ == '__main__':
    main(sys.argv)
