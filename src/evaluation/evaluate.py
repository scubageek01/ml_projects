import sys
import click
import datetime
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

sys.path.append('src')
from data.preprocess import read_processed_data


@click.command()
@click.option('--model', help='The model to be evaluated')
@click.option('--scoring', help='Scoring metrics i.e. accuracy, neg_log_loss, precision, etc..')
@click.option('--features', help='The feature matrix')
@click.option('--response', help='The response vector')
@click.argument('output_file', type=click.Path(writable=True, dir_okay=False))
def main(model, scoring, features, response, output_file):
    model = read_processed_data(model)
    X = read_processed_data(features)
    y = read_processed_data(response)
    kfold = KFold(n_splits=10, random_state=7)
    results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
    with open(output_file, 'a+') as fh:
        fh.write('Executed on {}\n\n'.format(datetime.datetime.now()))
        fh.write('Model parameters \n {}\n\n'.format(str(model.get_params())))
        fh.write('{}\n - Mean: {:.3f}  Std:  {:.3f}\n\n'.format(scoring.title(), results.mean(), results.std()))


if __name__ == '__main__':
   main()

