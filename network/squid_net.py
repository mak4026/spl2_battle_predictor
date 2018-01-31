import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from enum import Enum
from utils import dbs
from network.base_classifier import BaseClassifier

NetworkMode = Enum("NetworkMode", "NeuralNetwork logreg")


class SquidNetwork(BaseClassifier):
    def __init__(self, weapon_only=False, mode=NetworkMode.logreg):
        super().__init__(weapon_only)
        self.mode = mode
        self.name = mode.name

    def build_network(self):
        if self.weapon_only:
            idim = dbs.weapon_num * 2
        else:
            idim = dbs.weapon_num * 2\
                   + dbs.stage_num\
                   + dbs.rule_num\
                   + 1 # gachi power

        if self.mode is NetworkMode.NeuralNetwork:
            model = Sequential()
            model.add(Dense(20, input_dim=idim, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(1, activation='sigmoid'))
        elif self.mode is NetworkMode.logreg:
            model = Sequential()
            model.add(Dense(1, input_dim=idim, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def train(self, filename):
        all_x, all_y = self.load_data(filename)
        X_train, X_test, y_train, y_test = train_test_split(all_x, all_y, test_size=0.2)

        self.network = self.build_network()

        self.network.fit(X_train, y_train,
                         validation_split=0.25,
                         epochs=100, batch_size=12,
                         callbacks=[EarlyStopping()])

        score = self.network.evaluate(X_test, y_test)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
