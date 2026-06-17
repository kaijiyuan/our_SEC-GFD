# Revisiting Graph-based Fraud Detection in Sight of Heterophily and Spectrum (AAAI2024)

## Abstract
Graph-based fraud detection (GFD) can be regarded as a challenging semi-supervised node binary classification task. In recent years, Graph Neural Networks (GNN) have been widely applied to GFD, characterizing the anomalous possibility of a node by aggregating neighbor information. However, fraud graphs are inherently heterophilic, thus most of GNNs perform poorly due to their assumption of homophily. In addition, due to the existence of heterophily and class imbalance problem, the existing models do not fully utilize the precious node label information. To address the above issues, this paper proposes a semi-supervised GNN-based fraud detector SEC-GFD. This detector includes a hybrid filtering module and a local environmental constraint module, the two modules are utilized to solve heterophily and label utilization problem respectively. The first module starts from the perspective of the spectral domain, and solves the heterophily problem to a certain extent. Specifically, it divides the spectrum into various mixed-frequency bands based on the correlation between spectrum energy distribution and heterophily. Then in order to make full use of the node label information, a local environmental constraint module is adaptively designed. The comprehensive experimental results on four real-world fraud detection datasets denote that SEC-GFD outperforms other competitive graph-based fraud detectors.


## Implementation
The relevant datasets developed in the paper are on [google drive](https://drive.google.com/drive/folders/1eqfWN0CIudj7e9KJvkmj5uzK-eWs_pSE?usp=sharing). Download and unzip all files in the `data` folder.

This repository version uses the five DGL graph datasets under `data/all_data`:

- `questions`
- `reddit`
- `tolokers`
- `weibo`
- `yelp`

The training code uses split `0` from `train_masks`, `val_masks`, and `test_masks`. It also masks 99.5% of node feature entries with zeros before training. The missing-feature random seed is controlled by the same `--seed` argument used for training.

## Environment

The project is tested with Python 3.10 and the following core packages:

```text
torch==2.3.0
dgl==2.2.1
numpy
scipy
sympy
scikit-learn
```

Example Conda setup:

```bash
conda create -n sec-gfd python=3.10
conda activate sec-gfd
pip install torch==2.3.0
pip install dgl==2.2.1 numpy scipy sympy scikit-learn
```

For GPU training, install a CUDA-enabled DGL build that matches your CUDA and PyTorch versions. If DGL is CPU-only, run with `--device cpu`.

## Data Preparation

Place the dataset files in this structure:

```text
SEC-GFD/
  data/
    all_data/
      questions
      reddit
      tolokers
      weibo
      yelp
```

Each file should be a DGL graph file containing node fields:

```text
feature
label
train_masks
val_masks
test_masks
```

## Running

Run the default experiment:

```bash
python main.py
```

Run a specific dataset:

```bash
python main.py --dataset tolokers --epoch 100 --ntrials 1 --seed 0
```

Run on CPU:

```bash
python main.py --dataset tolokers --device cpu
```

Supported datasets:

```text
questions, reddit, tolokers, weibo, yelp
```

Important arguments:

```text
--dataset    Dataset name. Default: tolokers
--data_path  Dataset directory. Default: data/all_data
--epoch      Number of training epochs. Default: 100
--hid_dim    Hidden dimension. Default: 64
--order      Beta wavelet order. Default: 2
--lemda      Weight of the NCE loss. Default: 0.2
--device     cpu or cuda. Default: cuda
--seed       Random seed for training and feature masking. Default: 0
--ntrials    Number of repeated runs. Default: 1
```

The output metrics are:

```text
ROC-AUC
PR-AUC
Macro-F1
Fraud-F1
Fraud-Precision
Fraud-Recall
GMean
```

The current implementation keeps the original threshold-selection strategy based on the best test Macro-F1 over fixed thresholds.


If you use this package and find it useful, please cite our paper using the following BibTeX:)

```
@inproceedings{xu2024revisiting,
  title={Revisiting graph-based fraud detection in sight of heterophily and spectrum},
  author={Xu, Fan and Wang, Nan and Wu, Hao and Wen, Xuezhi and Zhao, Xibin and Wan, Hai},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  volume={38},
  number={8},
  pages={9214--9222},
  year={2024}
}
```
