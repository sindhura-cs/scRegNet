import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
import logging
import colorlog
import os
import json
import random
import sys

def set_seed(random_seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if is_dist():
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
    # if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)
    torch.cuda.manual_seed(random_seed)
    if is_dist():
        gpus = ",".join([str(_) for _ in range(int(os.environ["WORLD_SIZE"]))])
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

class RankFilter(logging.Filter):
    def filter(self, rec):
        return is_dist() == False or int(os.environ["RANK"]) == 0


class scRNADataset(Dataset):
    def __init__(self,train_set,num_gene,flag=False):
        super(scRNADataset, self).__init__()
        self.train_set = train_set
        self.num_gene = num_gene
        self.flag = flag

    def __getitem__(self, idx):
        train_data = self.train_set[:,:2]
        train_label = self.train_set[:,-1]

        if self.flag:
            train_len = len(train_label)
            train_tan = np.zeros([train_len,2])
            train_tan[:,0] = 1 - train_label
            train_tan[:,1] = train_label
            train_label = train_tan

        data = train_data[idx].astype(np.int64)
        label = train_label[idx].astype(np.float32)

        return data, label

    def __len__(self):
        return len(self.train_set)

    def Adj_Generate(self,TF_set,direction=False, loop=False):
        adj = sp.dok_matrix((self.num_gene, self.num_gene), dtype=np.float32)
        for pos in self.train_set:
            tf = pos[0]
            target = pos[1]

            if direction == False:
                if pos[-1] == 1:
                    adj[tf, target] = 1.0
                    adj[target, tf] = 1.0
            else:
                if pos[-1] == 1:
                    adj[tf, target] = 1.0
                    if target in TF_set:
                        adj[target, tf] = 1.0

        if loop:
            adj = adj + sp.identity(self.num_gene)

        adj = adj.todok()
        return adj



class load_data():
    def __init__(self, data, normalize=True):
        self.data = data
        self.normalize = normalize

    def data_normalize(self,data):
        standard = StandardScaler()
        epr = standard.fit_transform(data.T)
        return epr.T

    def exp_data(self):
        data_feature = self.data.values
        if self.normalize:
            data_feature = self.data_normalize(data_feature)
        data_feature = data_feature.astype(np.float32)
        return data_feature


def adj2saprse_tensor(adj):
    coo = adj.tocoo()
    i = torch.LongTensor([coo.row, coo.col])
    v = torch.from_numpy(coo.data).float()
    adj_sp_tensor = torch.sparse_coo_tensor(i, v, coo.shape)
    return adj_sp_tensor


def Evaluation(y_true, y_pred,flag=False):
    if flag:
        # y_p = torch.argmax(y_pred,dim=1)
        y_p = y_pred[:,-1]
        y_p = y_p.cpu().detach().numpy()
        y_p = y_p.flatten()
    else:
        y_p = y_pred.cpu().detach().numpy()
        y_p = y_p.flatten()

    y_t = y_true.cpu().numpy().flatten().astype(int)
    AUC = roc_auc_score(y_true=y_t, y_score=y_p)
    AUPR = average_precision_score(y_true=y_t,y_score=y_p)
    AUPR_norm = AUPR/np.mean(y_t)
    return AUC, AUPR, AUPR_norm


def normalize(expression):
    std = StandardScaler()
    epr = std.fit_transform(expression)
    return epr


def Network_Statistic(data_type,net_scale):
    dic = {'hESC500': 0.164, 'hESC1000': 0.165,'hHEP500': 0.379, 'hHEP1000': 0.377,'mDC500': 0.085,
            'mDC1000': 0.082,'mESC500': 0.345, 'mESC1000': 0.347,'mHSC-E500': 0.578, 'mHSC-E1000': 0.566,
            'mHSC-GM500': 0.543, 'mHSC-GM1000': 0.565,'mHSC-L500': 0.525, 'mHSC-L1000': 0.507}
    query = data_type + str(net_scale)
    scale = dic[query]
    return scale


def is_dist():
    return False if os.getenv("WORLD_SIZE") is None else True


def set_logging():
    root = logging.getLogger()
    # NOTE: clear the std::out handler first to avoid duplicated output
    if root.hasHandlers():
        root.handlers.clear()
    root.setLevel(logging.INFO)
    log_format = "[%(name)s %(asctime)s] %(message)s"
    color_format = "%(log_color)s" + log_format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(colorlog.ColoredFormatter(color_format))
    console_handler.addFilter(RankFilter())
    root.addHandler(console_handler)


def store_results(results, dir, data):
    metric_keys = results[0].keys()
    metrics = {key: np.array([result[key] for result in results]) for key in metric_keys}
    mean_metrics = {key: np.mean(metrics[key]) for key in metric_keys}
    std_metrics = {key: np.std(metrics[key]) for key in metric_keys}

    if data=='test':
        print(dir)
        print(mean_metrics)
        print(std_metrics)

    output = {
        'trials': results,  
        'mean_metrics': mean_metrics,  
        'std_metrics': std_metrics
    }

    # Save results to a file
    with open(os.path.join(dir, f'final_results_{data}.json'), "w") as f:
        json.dump(output, f, indent=4)

    print(f"Metrics and trials stored in final_results_{data}.json")