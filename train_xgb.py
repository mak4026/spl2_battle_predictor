from network.xgboost import XGBoostClassifier

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir',
                        help='前処理を行ったcsvファイルの場所を指定します',
                        type=str)
    args = parser.parse_args()

    xgb = XGBoostClassifier()
    xgb.train(args.data_dir)
