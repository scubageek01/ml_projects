import sys
import click
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR



PARAMS = 'params.yml'

sys.path.append('src')
from data.preprocess import read_processed_data, read_config, standardize_data, rescale_data


def get_estimators():
    estimators = []
    estimators.append(('LINREG', LinearRegression()))
    estimators.append(('RIDGE', Ridge()))
    estimators.append(('LASSO', Lasso()))
    estimators.append(('ELASTIC', ElasticNet()))
    estimators.append(('KNN', KNeighborsRegressor()))
    estimators.append(('CART', DecisionTreeRegressor()))
    estimators.append(('SVM', SVR()))
    return estimators


@click.command()
@click.option('--scoring', default='neg_mean_squared_error', help='Scoring metrics i.e. accuracy, neg_log_loss, precision, etc..')
@click.option('--features', default='data/processed/features.npy', help='The feature matrix')
@click.option('--response', default='data/processed/response.npy', help='The response vector')
@click.argument('output_file', default='reports/figures/model_comparison_results.txt', type=click.Path(writable=True, dir_okay=False))
def main(scoring, features, response, output_file):
    estimators = get_estimators()
    results = []
    names = []
    X_unscaled = read_processed_data(features)
    X = X_unscaled
    # X = rescale_data(X_unscaled)
    y = read_processed_data(response)
    kfold = KFold(n_splits=10, random_state=7)
    # Use cv = kfold if using kfold validation or cv = LeaveOneOut() for Leave One Out
    cv = kfold

    title = "Scoring metrics: {}\n\n".format(scoring)
    with open(output_file, 'a+') as fh:
        fh.write(title)

    for name, estimator in estimators:
        cv_results = cross_val_score(estimator, X, y, cv=cv, scoring=scoring)
        results.append(cv_results)
        names.append(name)

        msg = "{}:  Mean: {:.3f}  Std: {:.3f}\n".format(name, cv_results.mean(), cv_results.std())
        with open(output_file, 'a+') as fh:
            fh.write(msg)


if __name__ == '__main__':
    main()
