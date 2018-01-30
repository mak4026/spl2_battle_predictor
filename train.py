# from network.squid_net import SquidNetwork, NetworkMode
from network.svm import SVMClassifier
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir',
                        help='前処理を行ったcsvファイルの場所を指定します',
                        type=str)
    args = parser.parse_args()

    # network = SquidNetwork(mode=NetworkMode.logreg)
    # network.train(args.data_dir)

    svm = SVMClassifier()
    svm.train(args.data_dir)