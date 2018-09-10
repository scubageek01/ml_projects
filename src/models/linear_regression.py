import pickle
import sys

sys.path.append('src')

from sklearn.linear_model import LinearRegression
from data import get_features, get_response


class LinearRegressionModel(object):
    def __init__(self):
        self.name = 'Linear Regression'
        self.clf = LinearRegression()

    def get_params(self):
        return self.clf.get_params()

    def train(self, dataframe):
        X = get_features(dataframe)
        y = get_response(dataframe)
        self.clf.fit(X, y)

    def predict(self, X):
        y_pred = self.clf.predict(X)
        return y_pred

    def save(self, filename):
        with open(filename, 'wb') as output_file:
            pickle.dump(self.clf, output_file, pickle.HIGHEST_PROTOCOL)

    def load(self, filename):
        with open(filename, 'rb') as input_file:
            self.clf = pickle.load(input_file)
