import sys
import click
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

sys.path.append('src')
from data.preprocess import read_processed_data


@click.command()
@click.option('--model')
@click.option('--scoring')
@click.option('--features')
@click.option('--response')
@click.argument('output_file', type=click.Path(writable=True, dir_okay=False))
def main(model, scoring, features, response, output_file):
    model = read_processed_data(model)
    X = read_processed_data(features)
    y = read_processed_data(response)
    kfold = KFold(n_splits=10, random_state=7)
    results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
    with open(output_file, 'w') as fh:
        fh.write('{}\n - Mean: {:.3f}  Std:  {:.3f}\n\n'.format(scoring.title(), results.mean(), results.std()))
        fh.write('Model parameters \n {}\n\n'.format(str(model.get_params())))


if __name__ == '__main__':
   main()

