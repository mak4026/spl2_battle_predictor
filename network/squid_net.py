import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import StratifiedKFold
from enum import Enum
from utils import dbs

NetworkMode = Enum("NetworkMode", "simple logreg")

class SquidNetwork():
    def __init__(self, mode=NetworkMode.simple):
        self.mode = mode
        self.name = mode.name

    def build_network_simple(self):
        idim = dbs.weapon_num * 2\
               + dbs.stage_num\
               + dbs.rule_num\
               + 1 # gachi power

        if self.mode is NetworkMode.simple:
            model = Sequential()
            model.add(Dense(20, input_dim=idim, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(1, activation='sigmoid'))
        elif self.mode is NetworkMode.logreg:
            model = Sequential()
            model.add(Dense(1, input_dim=idim, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def load_data(self, filename):
        df = pd.read_csv(filename, header=None)
        df.dropna(inplace=True)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        return X, y

    def train(self, filename):
        all_x, all_y = self.load_data(filename)
        fold_num = 10
        skf = StratifiedKFold(n_splits=fold_num, shuffle=True)
        cvscores = []

        for train, test in skf.split(all_x, all_y):
            self.network = None
            self.network = self.build_network_simple()

            self.network.fit(all_x[train], all_y[train],
                             validation_split=0.33,
                             epochs=100, batch_size=8)
            scores = self.network.evaluate(all_x[test], all_y[test])
            print("%s: %.2f%%" % (self.network.metrics_names[1], scores[1]*100))
            cvscores.append(scores[1] * 100)
        print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
