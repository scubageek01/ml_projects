import click
import sys

sys.path.append('src')

import matplotlib.pyplot as plt
import seaborn as sns
from data.preprocess import read_raw_data, read_config

PARAMS = 'src/data/params.yml'


def pair_plot(dataframe, feature_cols, response_col):
    plot = sns.pairplot(dataframe, vars=feature_cols, hue=''.join(response_col))
    return plot


@click.command()
@click.argument('input_file', type=click.Path(exists=True, readable=True, dir_okay=False))
@click.argument('output_file', type=click.Path(writable=True, dir_okay=False))
def main(input_file, output_file):
    cfg = read_config(PARAMS)
    feature_cols = cfg['feature_cols']
    response_col = cfg['response_col']
    dataset_columns = feature_cols + response_col
    dataframe = read_raw_data(input_file)
    dataframe.columns = dataset_columns
    plot = pair_plot(dataframe, feature_cols, response_col)
    plot.savefig(output_file)


if __name__ == '__main__':
    main()

