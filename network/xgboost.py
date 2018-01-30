import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix


class XGBoostClassifier():
    def __init__(self):
        self.name = 'xgboost'

    def load_data(self, filename):
        df = pd.read_csv(filename, header=None)
        df.dropna(inplace=True)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        return X, y

    def train(self, filename):
        all_x, all_y = self.load_data(filename)

        X_train, X_test, y_train, y_test = train_test_split(all_x, all_y, test_size=0.2, random_state=0)

        # xgboostモデルの作成
        clf = xgb.XGBClassifier()

        # ハイパーパラメータ探索
        clf_cv = GridSearchCV(clf, {
            'max_depth': [2,4,6,8,10,12],
            'n_estimators': [50,100,200,300,500]
            }, verbose=2, n_jobs=6)
        clf_cv.fit(X_train, y_train)
        print(clf_cv.best_params_, clf_cv.best_score_)

        # 改めて最適パラメータで学習
        clf = xgb.XGBClassifier(**clf_cv.best_params_)
        clf.fit(X_train, y_train)

        # 学習モデルの保存、読み込み
        # import pickle
        # pickle.dump(clf, open("model.pkl", "wb"))
        # clf = pickle.load(open("model.pkl", "rb"))

        # 学習モデルの評価
        pred = clf.predict(X_test)
        print(confusion_matrix(y_test, pred))
        print(classification_report(y_test, pred))
