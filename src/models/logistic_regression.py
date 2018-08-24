import click
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import pickle


@click.command()
@click.argument('output_file', type=click.Path(writable=True, dir_okay=False))
@click.option('--scaler', default='standardized', help='standard -> StandardScaler, minmax -> MinMaxScaler')
def main(output_file, scaler='standardized'):
    estimators = []
    if scaler == 'standardized':
        estimators.append(('standardized', StandardScaler()))
    elif scaler == 'minmax':
        estimators.append(('minmax', MinMaxScaler()))
    else:
        pass
    estimators.append(('logreg', LogisticRegression()))
    model = Pipeline(estimators)

    with open(output_file, 'wb') as fh:
        pickle.dump(model, fh)


if __name__ == '__main__':
    main()
