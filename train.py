import sys
import warnings
import argparse
import numpy as np
import pandas as pd
import os
from data.data import process_data
from model import model
from keras.models import Model
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
    """Train a single model and save the trained model and loss history."""
    model.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
    hist = model.fit(
        X_train, y_train,
        batch_size=config["batch"],
        epochs=config["epochs"],
        validation_split=0.05)

    # Save model
    model_save_path = os.path.join(os.path.dirname(__file__), 'model', 'sites_models', f'{name}_{site}.h5')
    model.save(model_save_path)

    # Save training history
    loss_history_save_path = os.path.join(os.path.dirname(__file__), 'model', 'sites_models', f'{name}_{site}_loss.csv')
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv(loss_history_save_path, encoding='utf-8', index=False)


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
            models = model.get_saes([lag, 400, 400, 400, 1])
            m = models[-1]
            train_model(m, X_train, y_train, args.model, config, site)

        print(f"Finished training model for SCATS site: {site}")


if __name__ == '__main__':
    main(sys.argv)