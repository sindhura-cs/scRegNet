# scRegNet: Gene Regulatory Network Inference with Joint Representation from Graph Neural Network and Single-Cell Foundation Model

Sindhura Kommu, Yizhi Wang, Yue Wang and Xuan Wang

Abstract: *Inferring cell type-specific gene regulatory networks (GRNs) from single-cell RNA sequencing (scRNA-seq) data is a complex task, primarily due to data sparsity, noise, and the dynamic, context-dependent nature of gene regulation across cell types and states. Recent advancements in the collection of experimentally validated data on transcription factor binding have facilitated GRN inference via supervised machine learning methods. However, these methods still face challenges in 1) effectively representing and integrating prior knowledge, and 2) capturing regulatory mechanisms across diverse cellular contexts. To tackle the above challenges, we introduce a novel GRN inference method, scRegNet, that learns a joint representation from graph neural networks (GNNs) and pre-trained single-cell foundation models (scFMs). scRegNet combines rich contextual representations learned by large-scale, single-cell foundation models—trained on extensive unlabeled scRNA-seq datasets—with the structured knowledge embedded in experimentally validated networks through GNNs. This integration enables robust inference by simultaneously accounting for gene expression patterns and established gene regulatory networks. We evaluated our approach on seven single-cell scRNA-seq benchmark datasets from the BEELINE study outperforming current state-of-the-art methods in cell-type-specific GRN inference. scRegNet demonstrates a superior ability to capture intricate regulatory interactions between genes across various cell types, providing a more in-depth understanding of cellular processes and regulatory dynamics. By harnessing the capabilities of large-scale pre-trained single-cell foundation models and GNNs, scRegNet offers a scalable and adaptable tool for advancing research in cell type-specific gene interactions and biological functions.*

## Installation
```bash
pip install -r requirements.txt
```

[Optional] We recommend using [wandb](https://wandb.ai/) for logging and visualization.

```bash
pip install wandb
```

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