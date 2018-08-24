import click
import pandas as pd
import numpy as np
import yaml
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer

PROJ_ROOT = os.path.abspath(os.path.join(os.pardir))
PARAMS = 'src/data/params.yml'


def read_config(filename):
    with open(filename, 'r') as fh:
        config = yaml.load(fh)
    return config


def get_dataset_attributes():
    feature_cols = read_config('src/data/params.yml')['feature_cols']
    response_col = read_config('src/data/params.yml')['response_col']
    colnames = feature_cols + response_col
    feature_size = len(feature_cols)
    return colnames, feature_cols, response_col, feature_size


def read_raw_data(filename, names=None, header=None):
    dataframe = pd.read_csv(filename, names=names, header=header)
    return dataframe


def preprocess_data(dataframe):
    cfg = read_config(PARAMS)
    dataframe = dataframe.copy()
    dataframe.columns = cfg['feature_cols'] + cfg['response_col']
    return dataframe


def read_processed_data(filename):
    dataframe = np.load(open(filename, 'rb'))
    return dataframe


def rescale_data(dataframe):
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataframe = scaler.fit_transform(dataframe)
    return dataframe


def standardize_data(dataframe):
    scaler = StandardScaler()
    dataframe = scaler.fit_transform(dataframe)
    return dataframe


def normalize_data(dataframe):
    scaler = Normalizer()
    dataframe = scaler.fit_transform(dataframe)
    return dataframe


def binarize_data(dataframe):
    scaler = Binarizer()
    dataframe = scaler.fit_transform(dataframe)
    return dataframe


@click.command()
@click.argument('input_file', type=click.Path(exists=True, readable=True, dir_okay=False))
@click.argument('output_file', type=click.Path(writable=True, dir_okay=False))
@click.option('--features')
@click.option('--response')
def main(input_file, output_file, features, response):
    """Need to include logging feature.  But for now just print to console"""
    print("Preprocess data")

    dataframe = read_raw_data(input_file)
    colnames, feature_cols, response_col, feature_size = get_dataset_attributes()
    dataframe.columns = colnames
    dataframe.to_pickle(output_file)
    features_matrix = dataframe[feature_cols]
    response_vector = dataframe[response_col]
    features_matrix.to_pickle(features)
    response_vector.to_pickle(response)


if __name__ == '__main__':
    main()

