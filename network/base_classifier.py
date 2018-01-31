import numpy as np
import pandas as pd
from utils.dbs import weapon_num

class BaseClassifier():
    def __init__(self, weapon_only=False):
        self.weapon_only = weapon_only

    def train(self):
        pass

    def load_data(self, filename):
        df = pd.read_csv(filename, header=None)
        df.dropna(inplace=True)
        df = self.undersampling(df)

        if self.weapon_only:
            print("weapon feature only mode.")
            X = df.iloc[:, :weapon_num*2]
        else:
            X = df.iloc[:, :-1]

        X = X.values
        y = df.iloc[:, -1].values
        return X, y

    def undersampling(self, df):
        result_col = df.columns[-1]
        counts_dict = dict(df[result_col].value_counts())
        low_result, low_sample_num = min(counts_dict.items())

        low_freq_data = df[df[result_col] == low_result]

        high_freq_data_idx = df[df[result_col] == (1.0 - low_result)].index
        random_indices = np.random.choice(high_freq_data_idx, low_sample_num, replace=False)

        high_frequentry_data_sampled = df.loc[random_indices]

        merged_data = pd.concat([low_freq_data, high_frequentry_data_sampled], ignore_index=True)
        return merged_data.sample(frac=1).reset_index(drop=True)
