import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import gc
import logging
import random
import torch
from args import load_args
from utils import set_logging, store_results
import warnings
from models import scTransNet_GCN, scTransNet_SAGE, scTransNet_GAT
from utils import scRNADataset, load_data, adj2saprse_tensor
from sklearn.metrics import roc_auc_score, average_precision_score
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


def Evaluation(y_true, y_pred, flag=False):
    if flag:
        y_p = y_pred[:, -1]
        y_p = y_p.cpu().detach().numpy()
        y_p = y_p.flatten()
    else:
        y_p = y_pred.cpu().detach().numpy()
        y_p = y_p.flatten()

    y_t = y_true.cpu().numpy().flatten().astype(int)

    AUC = roc_auc_score(y_true=y_t, y_score=y_p)
    AUPR = average_precision_score(y_true=y_t, y_score=y_p)

    return {
        'AUC': AUC,
        'AUPR': AUPR
    }


class Infer: 
    def __init__(self, args, **kwargs):
        self.args = args
        self.trial = kwargs.pop("trial", None)
    
    @property
    def device(self):
        return torch.device(self.args.single_gpu if torch.cuda.is_available() else "cpu")
    
    def _get_embeddings(self, gene_num, data_input):
        if self.args.llm_type == "Geneformer":
            scFM_embs = os.path.join(self.args.scFM_folder, "Geneformer")
            embs = pd.read_csv(os.path.join(scFM_embs, f'{self.args.cell_type}_{self.args.num_TF}_gene_embeddings.csv'))
            final_df = pd.read_csv(os.path.join(scFM_embs, f'{self.args.cell_type}_{self.args.num_TF}.csv'))
            
            x = np.zeros((len(embs.columns)-1))
            gene_embeddings = []
            for _, row in final_df.iterrows():
                ensembl_id = row['ensembl_id']
                if ensembl_id in embs['Unnamed: 0'].values:
                    gene_emb = embs[embs['Unnamed: 0']==ensembl_id].values[0][1:]
                    gene_embeddings.append(np.array(gene_emb, dtype=np.float32))
                else:
                    gene_embeddings.append(x)

            gene_embeddings = np.array(gene_embeddings)

        elif self.args.llm_type == "scBERT":
            scFM_embs = os.path.join(self.args.scFM_folder, "scBERT")
            cell_embeddings_arr = np.load(os.path.join(scFM_embs, f'{self.args.cell_type}_{self.args.num_TF}_cell_embeddings.npy'))
            gene_embeddings = np.mean(cell_embeddings_arr, axis=0)

        elif self.args.llm_type == "scFoundation":
            scFM_embs = os.path.join(self.args.scFM_folder, "scFoundation")
            gene_list_df = pd.read_csv(os.path.join(scFM_embs, 'OS_scRNA_gene_index.19264.tsv'), header=0, delimiter='\t')
            gene_list = list(gene_list_df['gene_name'])
            x = np.zeros(512)

            gene_embeddings = np.load(os.path.join(scFM_embs, f'genemodule_{self.args.cell_type}_{self.args.num_TF}_singlecell_gene_embedding_f2_resolution.npy'))
            pooled_gene_embeddings = np.mean(gene_embeddings, axis=0)
            
            final_gene_embeddings = []
            cnt = 0
            for i in data_input.index:
                try:
                    final_gene_embeddings.append(pooled_gene_embeddings[gene_list.index(i)])
                except:
                    final_gene_embeddings.append(x)
                    cnt=cnt+1
                    
            gene_embeddings = np.array(final_gene_embeddings)
        
        feature1 = torch.from_numpy(gene_embeddings).float()
        return feature1
        
   
    def _prepare_data(self):
        path = os.path.join(self.args.data_folder, f'{self.args.cell_type}/TFs+{self.args.num_TF}/')
        exp_file = os.path.join(path, 'BL--ExpressionData.csv')
        train_file = os.path.join(path, 'Train_set.csv')
        test_file = os.path.join(path, 'Test_set.csv')
        tf_file = os.path.join(path, 'TF.csv')
        target_file = os.path.join(path, 'Target.csv')
        data_input = pd.read_csv(exp_file, index_col=0)
        train_data = pd.read_csv(train_file, index_col=0).values
        test_data = pd.read_csv(test_file, index_col=0).values
        tf = pd.read_csv(tf_file,index_col=0)['index'].values.astype(np.int64)
        tf = torch.from_numpy(tf).to(self.device)
        target = pd.read_csv(target_file,index_col=0)['index'].values.astype(np.int64)
        target = torch.from_numpy(target).to(self.device)

        loader = load_data(data_input)
        feature2 = loader.exp_data()
        feature2 = torch.from_numpy(feature2)
        gene_num = feature2.shape[0]
        feature1 = self._get_embeddings(gene_num, data_input)
        self.input_dim = feature2.size()[1]
        self.gene_dim = feature1.size()[1]

        data_feature2 = feature2.to(self.device)
        if self.args.llm_type == "scBERT":
            feature1 = feature1[:-1]
        
        data_feature1 = feature1.to(self.device)
        train_load = scRNADataset(train_data, gene_num, flag=self.args.flag)
        adj = train_load.Adj_Generate(tf, loop=self.args.loop)
        adj = adj2saprse_tensor(adj)
        adj = adj.to(self.device)
        train_data = torch.from_numpy(train_data)
        train_data = train_data.to(self.device)
        test_data = torch.from_numpy(test_data)
        test_data = test_data.to(self.device)

        return train_data, test_data, adj, data_feature1, data_feature2
    
    def get_model(self):

        if self.args.gnn_type == "GCN":
            model = scTransNet_GCN(input_dim=self.input_dim,
                                   args=self.args,
                                   gene_dim=self.gene_dim,
                                   device=self.device
                                   ).to(self.device)
        elif self.args.gnn_type == "GraphSAGE":
            model = scTransNet_SAGE(input_dim=self.input_dim,
                                   args=self.args,
                                   gene_dim=self.gene_dim,
                                   device=self.device
                                   ).to(self.device)
        elif self.args.gnn_type == "GAT":
            model = scTransNet_GAT(input_dim=self.input_dim,
                                   args=self.args,
                                   gene_dim=self.gene_dim,
                                   device=self.device
                                   ).to(self.device)
        return model
    
    
    def infer(self):
        train_data, test_data, adj, data_feature1, data_feature2 = self._prepare_data()
        self.model = self.get_model()
        model_path = os.path.join(self.args.output_dir, f"best/ckpt/model_seed{self.args.random_seed}.pt")
        self.model.load_state_dict(torch.load(model_path)) 
        self.model.eval()

        results_train = []
        results_test = []
        for _ in tqdm(range(50)):
            score_test = self.model(data_feature2, adj, test_data, data_feature1)
            score_train = self.model(data_feature2, adj, train_data, data_feature1)

            if self.args.flag:
                score_train = torch.softmax(score_train, dim=1)
                score_test = torch.softmax(score_test, dim=1)
            else:
                score_train = torch.sigmoid(score_train)
                score_test = torch.softmax(score_test, dim=1)
        
            metrics_train = Evaluation(y_pred=score_train, y_true=train_data[:, -1],flag=self.args.flag)
            metrics_test = Evaluation(y_pred=score_test, y_true=test_data[:, -1],flag=self.args.flag)

            results_train.append(metrics_train)
            results_test.append(metrics_test)

        return results_train, results_test


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
    best_dir = './out/GCN/Geneformer/tf_1000_mHSC-L/without_scFM/best/'
    results_train, results_test = main(best_dir)
    store_results(results_train, best_dir, 'train')
    store_results(results_test, best_dir, 'test')