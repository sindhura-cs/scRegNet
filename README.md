# scRegNet: Prediction of Gene Regulatory Connections with Joint Single-Cell Foundation Models and Graph-Based Learning

We provide PyTorch implementation for scRegNet that combines single-cell foundation models and graph-based learning to predict gene regulatory connections.

<p align="center">
  <img src="./figs/Overview.pdf" width="1000" title="scRegNet framework overview" alt="">
</p>

## Links

- [Installation](#installation)
- [Data](#data)
- [Download pretrained weights](#download-pretrained-weights)
- [Train](#train)
- [Inference](#inference)
- [Acknowledgment](#acknowledgment)

## Installation

For training, a GPU is strongly recommended.

#### PyTorch

The code is based on PyTorch. You can find installation instructions [here](http://pytorch.org/).

#### Dependencies
* Python == 3.10
* PyTorch == 2.4.1
* scikit-learn == 1.5.2
* numpy == 1.20.3
* optuna == 4.0.0

[Optional] We recommend using [wandb](https://wandb.ai/) for logging and visualization.

```bash
pip install wandb
```
**Note: PyTorch 2.4.1 and CUDA 12.4 were used during development.**

## Data

We use seven publicly available scRNA-seq benchmark datasets by BEELINE (Pratapa et al., 2020) for gene regulatory link prediction from single-cell transcriptomic data.

## Download pretrained weights

## Train

## Inference

## Usage
```bash
$ git clone this-repo-url
$ cd scRegNet
$ bash gnn_hp.sh tf_500_hESC GCN hESC 500 Geneformer
```

## Acknowledgements

We sincerely thank the authors of following open-source projects:

- [Geneformer](https://huggingface.co/ctheodoris/Geneformer)
- [scFoundation](https://github.com/biomap-research/scFoundation)
- [scBERT](https://github.com/TencentAILabHealthcare/scBERT)
- [Optuna](https://github.com/optuna/optuna)
- [BEELINE](https://github.com/Murali-group/Beeline)
