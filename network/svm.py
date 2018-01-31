import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.svm import SVC
from utils import dbs
from network.base_classifier import BaseClassifier


class SVMClassifier(BaseClassifier):
    def __init__(self, weapon_only=False):
        super().__init__(weapon_only)
        self.name = 'SVM'

    def train(self, filename):
        all_x, all_y = self.load_data(filename)

        X_train, X_test, y_train, y_test = train_test_split(all_x, all_y, test_size=0.2)
        tuned_parameters = [
            # {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
            {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.001, 0.0005, 0.0001]},
            # {'C': [1, 10, 100, 1000], 'kernel': ['poly'], 'degree': [2, 3, 4], 'gamma': [0.001, 0.0001]},
            {'C': [1, 10, 100, 1000], 'kernel': ['sigmoid'], 'gamma': [0.001, 0.0001]}
            ]
        clf = GridSearchCV(
            SVC(), # 識別器
            tuned_parameters, # 最適化したいパラメータセット
            cv=5, # 交差検定の回数
            scoring='f1',
            verbose=2,
            n_jobs=6
            ) # モデルの評価関数の指定
        clf.fit(X_train, y_train)


        print("# Tuning hyper-parameters for f1")
        print()
        print("Best parameters set found on development set: %s" % clf.best_params_)
        print()

        # それぞれのパラメータでの試行結果の表示
        print("Grid scores on development set:")
        print()
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean_score, scores.std() * 2, params))
        print()

        # テストデータセットでの分類精度を表示
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print("test accuracy: {}".format(accuracy_score(y_test, y_pred)))
        print("confusion_matrix:")
        print(confusion_matrix(y_test, y_pred))
