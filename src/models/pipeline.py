import click
import sys
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

PARAMS = 'params.yml'

sys.path.append('src')
from data.preprocess import read_processed_data, read_config


@click.command()
@click.option('--features', default='data/processed/features.npy', help='The feature matrix')
@click.option('--response', default='data/processed/response.npy', help='The response vector')
def main(features, response):
    X = read_processed_data(features)
    y = read_processed_data(response)
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('LR', LinearDiscriminantAnalysis()))
    model = Pipeline(estimators)
    kfold = KFold(n_splits=10, random_state=7)
    cv = kfold
    results = cross_val_score(model, X, y, cv=cv)
    print(results.mean())


if __name__ == '__main__':
    main()