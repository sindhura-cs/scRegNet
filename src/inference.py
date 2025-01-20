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

    infer = Infer(args)
    results_train, results_test = infer.infer()

    del infer
    torch.cuda.empty_cache()
    gc.collect()
    return results_train, results_test

if __name__ == "__main__":
    best_dir = './out/GCN/scBERT/tf_1000_hESC/best/'
    results_train, results_test = main(best_dir)
    store_results(results_train, best_dir, 'train')
    store_results(results_test, best_dir, 'test')