if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir',
                        help='前処理を行ったcsvファイルの場所を指定します',
                        type=str)
    parser.add_argument('--weapon_only', '-w',
                        help='ブキ編成のみを特徴量として用いて分類します',
                        action='store_true')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--logreg',
                        help='ロジスティック回帰で分類します(デフォルト)',
                        action='store_true')
    group.add_argument('--nn',
                        help='ニューラルネットワークで分類します',
                        action='store_true')
    group.add_argument('--svm',
                        help='SVMで分類します',
                        action='store_true')
    group.add_argument('--xgb',
                        help='xgboostで分類します',
                        action='store_true')

    args = parser.parse_args()

    if args.nn:
        from network.squid_net import SquidNetwork, NetworkMode
        network = SquidNetwork(weapon_only=args.weapon_only, mode=NetworkMode.NeuralNetwork)
    elif args.svm:
        from network.svm import SVMClassifier
        network = SVMClassifier(weapon_only=args.weapon_only)
    elif args.xgb:
        from network.xgboost import XGBoostClassifier
        network = XGBoostClassifier(weapon_only=args.weapon_only)
    else:
        from network.squid_net import SquidNetwork, NetworkMode
        network = SquidNetwork(weapon_only=args.weapon_only, mode=NetworkMode.logreg)

    print(network.name)
    network.train(args.data_dir)

    # svm = SVMClassifier()
    # svm.train(args.data_dir)
