import sys
import click
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

PARAMS = 'params.yml'

sys.path.append('src')
from data.preprocess import read_processed_data, read_config, standardize_data, rescale_data


def get_estimators():
    estimators = []
    estimators.append(('LR', LogisticRegression()))
    estimators.append(('LDA', LinearDiscriminantAnalysis()))
    estimators.append(('KNN', KNeighborsClassifier()))
    estimators.append(('CART', DecisionTreeClassifier()))
    estimators.append(('RFC', RandomForestClassifier()))
    estimators.append(('NB', GaussianNB()))
    estimators.append(('SVM', SVC()))
    return estimators


@click.command()
@click.option('--scoring', default='accuracy', help='Scoring metrics i.e. accuracy, neg_log_loss, precision, etc..')
@click.option('--features', default='data/processed/features.npy', help='The feature matrix')
@click.option('--response', default='data/processed/response.npy', help='The response vector')
@click.argument('output_file', default='reports/figures/model_comparison_results.txt', type=click.Path(writable=True, dir_okay=False))
def main(scoring, features, response, output_file):
    estimators = get_estimators()
    results = []
    names = []

    title = "Scoring metrics: {}\n\n".format(scoring)
    X = read_processed_data(features)
    y = read_processed_data(response)
    kfold = KFold(n_splits=10, random_state=7)
    cv = kfold  # or cv=LeaveOneOut()

    with open(output_file, 'a+') as fh:
        fh.write(title)

    for name, estimator in estimators:
        cv_results = cross_val_score(estimator, X, y, cv=cv, scoring=scoring)
        results.append(cv_results)
        names.append(name)

        y_pred = cross_val_predict(estimator, X, y, cv=cv)
        cls_rpt = classification_report(y, y_pred)
        conf_mat = confusion_matrix(y, y_pred)
        conf_mat = np.array2string(conf_mat)

        msg = "{}:  Mean: {:.3f}  Std: {:.3f}\n".format(name, cv_results.mean(), cv_results.std())
        with open(output_file, 'a+') as fh:
            fh.write(msg)
            fh.write('\n')
            fh.write(conf_mat)
            fh.write('\n')
            fh.write(cls_rpt)
            fh.write('\n')


if __name__ == '__main__':
    main()
