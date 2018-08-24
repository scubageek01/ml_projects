import click
import pandas as pd
import numpy as np
import yaml
import os

PROJ_ROOT = os.path.abspath(os.path.join(os.pardir))
PARAMS = 'src/data/params.yml'


def read_config(filename):
    with open(filename, 'r') as fh:
        config = yaml.load(fh)
    return config


def get_dataset_attributes():
    feature_cols = read_config(PARAMS)['feature_cols']
    response_col = read_config(PARAMS)['response_col']
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


def get_features(dataframe):
    colnames, feature_cols, response_col, feature_size = get_dataset_attributes()
    return dataframe[feature_cols]


def get_response(dataframe):
    colnames, feature_cols, response_col, feature_size = get_dataset_attributes()
    return dataframe[response_col]


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

    features_matrix = get_features(dataframe)
    response_vector = get_response(dataframe)

    features_matrix.to_pickle(features)
    response_vector.to_pickle(response)


if __name__ == '__main__':
    main()

