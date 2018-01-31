import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from network.base_classifier import BaseClassifier


class XGBoostClassifier(BaseClassifier):
    def __init__(self, weapon_only=False):
        super().__init__(weapon_only)
        self.name = 'xgboost'

    def train(self, filename):
        all_x, all_y = self.load_data(filename)

        X_train, X_test, y_train, y_test = train_test_split(all_x, all_y, test_size=0.2)

        # xgboostモデルの作成
        xgb_clf = xgb.XGBClassifier(objective='binary:logistic')

        # ハイパーパラメータ探索
        clf = GridSearchCV(
            xgb_clf, {
                'max_depth': [2, 4, 6],
                'n_estimators': [25, 50, 100, 200]
            },
            cv=5,
            scoring='f1',
            verbose=2,
            n_jobs=6)
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
        # 学習モデルの保存、読み込み
        # import pickle
        # pickle.dump(clf, open("model.pkl", "wb"))
        # clf = pickle.load(open("model.pkl", "rb"))

