import click
import sys
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.externals import joblib

PARAMS = 'params.yml'

sys.path.append('src')
from data.preprocess import read_processed_data, read_config


@click.command()
@click.argument('output_file', type=click.Path(writable=True, dir_okay=False))
@click.option('--scaler', default='standardized', help='standardized -> StandardScaler, minmax -> MinMaxScaler')
@click.option('--features', default='data/processed/features.npy', help='The feature matrix')
@click.option('--response', default='data/processed/response.npy', help='The response vector')
def main(output_file, features, response,  scaler='standardized'):
    X = read_processed_data(features)
    y = read_processed_data(response)
    estimators = []
    if scaler == 'standardized':
        estimators.append(('standardized', StandardScaler()))
    elif scaler == 'minmax':
        estimators.append(('minmax', MinMaxScaler()))
    else:
        pass
    estimators.append(('lda', LinearDiscriminantAnalysis()))
    model = Pipeline(estimators)
    model.fit(X, y)

    # Evaluate pipeline
    kfold = KFold(n_splits=10, random_state=7)
    cv = kfold
    results = cross_val_score(model, X, y, cv=cv)
    print(results.mean())

    joblib.dump(model, output_file)

    test_data = read_processed_data('data/processed/features.npy')[0:5,]

    print(model.predict(test_data))


if __name__ == '__main__':
    main()
