# -*_ coding: utf-8 -*-
r"""
    neural network by sklearn
"""
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from utilities.load_data import load_data, split_features_labels


# from ..rl_issues import *


class NN(object):

    def __init__(self, input_file=''):
        self.dataset = load_data(input_file)
        self.model = MLPClassifier(max_iter=2000)
        self.model = DecisionTreeClassifier()
        self.model = XGBClassifier(n_estimators=1000, max_depth=100, learning_rate=0.01)

    def train(self, train_data):
        X, y = train_data
        self.model.fit(X, y)

    def evaluate(self, test_data):
        X, y = test_data
        y_preds = self.model.predict(X)
        cm = confusion_matrix(y, y_preds)
        print(f"cm={cm}")
        acc = ((y_preds == y) * 1).sum() / len(y)
        print(f"acc={acc}")
        clf_rpt = classification_report(y, y_preds)
        print(f"{clf_rpt}")


def main():
    input_file = '../input_data/iris.csv'
    data = load_data(input_file)
    X, y = split_features_labels(data)
    y = np.asarray(y, dtype=int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    nn_mdl = NN(input_file)
    nn_mdl.train((X_train, y_train))

    nn_mdl.evaluate((X_train, y_train))
    nn_mdl.evaluate((X_test, y_test))


if __name__ == '__main__':
    main()
