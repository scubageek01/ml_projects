import click
import sys
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

sys.path.append('src')

from data.preprocess import read_processed_data


@click.command()
@click.option('--features', default='data/processed/features.npy', help='The feature matrix')
@click.option('--response', default='data/processed/response.npy', help='The response vector')
@click.option('--output_tuning_results', default='reports/figures/tuning_results.txt', help='Results of tuning')
@click.option('--output_tuned_model', default='models/knn.model')
def tune_knn(features, response, output_tuning_results, output_tuned_model):
    estimator = KNeighborsClassifier()
    n_neighbor_range = list(range(1, 31))
    weights = ['uniform', 'distance']
    param_grid = dict(n_neighbors=n_neighbor_range, weights=weights)
    grid = GridSearchCV(estimator, param_grid, cv=10, scoring='accuracy')
    X = read_processed_data(features)
    y = read_processed_data(response)
    grid.fit(X, y)
    tuning_results = 'Best score: {}\nBest estimator: {}\nBest params: {}\n'.format(grid.best_score_,
                                                                                    grid.best_params_,
                                                                                    grid.best_estimator_)
    with open(output_tuning_results, 'w') as fh:
        fh.write(tuning_results)
    joblib.dump(grid.best_estimator_, output_tuned_model)


if __name__ == '__main__':
    tune_knn()
