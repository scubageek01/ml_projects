import click
import sys
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

sys.path.append('src')

from data.preprocess import read_processed_data, get_dataset_attributes


@click.command()
@click.option('--model')
@click.option('--scoring')
@click.argument('input_file', type=click.Path(exists=True, readable=True, dir_okay=False))
@click.argument('output_file', type=click.Path(writable=True, dir_okay=False))
def main(model, input_file, scoring, output_file):
    dataframe = read_processed_data(input_file)
    model = read_processed_data(model)
    colnames, feature_cols, response_col, size = get_dataset_attributes()
    X = dataframe[feature_cols]
    y = dataframe[response_col]
    kfold = KFold(n_splits=10, random_state=7)
    results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
    print('Accuracy - Mean: {:.3f}  Std:  {:.3f}'.format(results.mean(), results.std()))


if __name__ == '__main__':
   main()

