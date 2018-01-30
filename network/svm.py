import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from utils import dbs

class SVMClassifier():
    def __init__(self):
        self.name = 'SVM'

    def load_data(self, filename):
        df = pd.read_csv(filename, header=None)
        df.dropna(inplace=True)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        return X, y

    def train(self, filename):
        all_x, all_y = self.load_data(filename)

        X_train, X_test, y_train, y_test = train_test_split(all_x, all_y, test_size=0.2, random_state=0)
        tuned_parameters = [
            # {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
            {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.001, 0.0005]},
            # {'C': [1, 10, 100, 1000], 'kernel': ['poly'], 'degree': [2, 3, 4], 'gamma': [0.001, 0.0001]},
            # {'C': [1, 10, 100, 1000], 'kernel': ['sigmoid'], 'gamma': [0.001, 0.0001]}
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
        print(clf.grid_scores_)
        # print('hoge')
        # clf = SVC()
        # clf.fit(X_train, y_train)
        # test_pred = clf.predict(X_test)

        # print(classification_report(y_test, test_pred))
        # print(accuracy_score(y_test, test_pred))
