# splatoon2 battle result predictor
Splatoon2の戦績データから，バトルの勝敗を学習し，予測します．

## Requirements
- Python3.5+
- pip
- numpy
- pandas
- scikit-learn
- Keras, tensorflow (`--logreg, --nn`使用時)
- XGBoost (`--xgb`使用時)

## Usage
### データを集める
<https://stat.ink>さんからデータを **迷惑がかからないように** 集めてください.
API doc: <https://github.com/fetus-hina/stat.ink/tree/master/doc/api-2>

### データの前処理

```bash
$ python3 utils/data_preparation.py data_dir [--start START] [--end END] --dst DST
```

- `data_dir`: 前処理を行いたい，stat.ink形式のjsonファイルが入ってるディレクトリを指定します．
- `--start`, `-s`: 前処理の対象とする期間の先頭の日付をYYYY-mm-dd-hhで指定します.
- `--end`, `-e`: 前処理の対象とする期間の終端の日付をYYYY-mm-dd-hhで指定します.
- `--dst`, `-d`: 前処理を行った後のCSVファイルの出力先を指定します．

for example (extract battles in v2.1.x)

```bash
$ python3 utils/data_preparation.py hoge/ -s 2017-12-13-10 -e 2018-1-17-10 -d csvs/v21x.csv
...
Extracted 24009 battles.
$ ls csvs/
v21x.csv
```

### 学習
```bash
$ python3 train.py data_dir [--weapon_only] [--logreg | --nn | --svm | --xgb]
```

- `data_dir`: 学習に使う前処理が完了しているデータ(CSVファイル)を指定します．
- `--weapon_only`または`-w` をつけると，ブキ編成のみのデータから学習を行います.

以下は学習に使う分類器を指定するフラグです．何も指定しないと，`--logreg`をつけるのと同じです．
- `--logreg`: ロジスティック回帰で学習を行います．
- `--nn`: （単純な）ニューラルネットワークで学習を行います．
- `--svm`: SVMで学習を行います．(NOTE: とても時間がかかります．)
- `--xgb`: XGBoostで学習を行います．

for example (weapon_only, using SVM)
```bash
$ python3 train.py csvs/v21x.csv -w --svm
```
