import numpy as np
import scipy.sparse as sp
import scanpy as sc
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc
from IPython.display import display
import sys
import os
import gc
import argparse
import json
import logging
import random
import math
import random
from functools import reduce
from scipy import sparse
import scipy.sparse as sp
import scipy.io as sio
from sklearn.metrics import recall_score, confusion_matrix, precision_recall_fscore_support, classification_report
import torch
from torch import nn
from torch.optim import Adam, SGD, AdamW
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CyclicLR
from args import parse_args, save_args, load_args
from infer import Infer
from utils import set_logging, store_results
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


def set_seed(random_seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)


def main(best_dir):
    set_logging()
    
    args = load_args(os.path.join(best_dir, 'ckpt'))
    logger.info(args)

    # args.random_seed = args.start_seed
    # set_seed(random_seed=args.random_seed)

    # args.gnn_lr = 0.0003362171651078414
    # args.gnn_weight_decay = 1.3352649486036125e-05
    # args.gnn_dropout = 0.6138338762768143
    # args.gnn_num_layers = 1
    # args.mlp_num_layers = 1
    # args.batch_size = 73
    # args.gnn_hidden_dims = [68]  
    # args.mlp_hidden_dims = [42] 
    # args.optimizer_name = 'Adam'

    # args.dataset = 'tf_1000_mHSCL'
    # args.model_type = 'GCN'
    # args.llm_type = 'geneformer'
    # args.cell_type = 'mHSC-L'
    # args.cell_t = 'mHSCL'
    # args.num_TF = '1000'
    # args.species = 'mouse'
    # args.suffix = 'optuna'
    # args.single_gpu = 1

    # args.output_dir = f"/data/wang/sindhura/Geneformer_prev/examples/out/{args.dataset}/{args.model_type}/{args.suffix}"
    # args.ckpt_dir = f"{args.output_dir}/ckpt"
    # best_output_dir = os.path.join(args.output_dir, "best")
    
    # save_args(args, best_output_dir)
    infer = Infer(args)
    results_train, results_test = infer.infer()

    del infer
    torch.cuda.empty_cache()
    gc.collect()
    return results_train, results_test


if __name__ == "__main__":
    best_dir = './out/GCN/Geneformer/tf_500_hESC/'
    results_train, results_test = main(best_dir)
    store_results(results_train, best_dir, 'train')
    store_results(results_test, best_dir, 'test')