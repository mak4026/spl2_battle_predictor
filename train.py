from network.squid_net import SquidNetwork
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir',
                        help='前処理を行ったcsvファイルの場所を指定します',
                        type=str)
    args = parser.parse_args()

    network = SquidNetwork()
    network.train(args.data_dir)