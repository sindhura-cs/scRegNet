import numpy as np
import pandas as pd
import os
import gc
import logging
import torch
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.models import scTransNet_GCN, scTransNet_SAGE, scTransNet_GAT
from src.utils import scRNADataset, load_data, adj2saprse_tensor, Evaluation
from src.utils import set_logging, set_seed
from src.args import parse_args, save_args
import warnings

from optuna.exceptions import ExperimentalWarning

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=ExperimentalWarning, module="optuna.multi_objective")
warnings.filterwarnings("ignore", category=FutureWarning)


class Trainer: 
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
            SEQ_LEN = gene_num + 1
            scFM_embs = os.path.join(self.args.scFM_folder, "scBERT")
            cell_embeddings_arr = np.load(os.path.join(scFM_embs, f'{self.args.cell_type}_{self.args.num_TF}_cell_embeddings.npy'))
            attn_scores_arr = np.load(os.path.join(scFM_embs, f'{self.args.cell_type}_{self.args.num_TF}_attention_maps.npy'))
            attn_scores_genes = attn_scores_arr.mean(1)
            attn_scores_norm = attn_scores_genes / np.sum(attn_scores_genes, axis=1, keepdims=True)

            num_cells = cell_embeddings_arr.shape[0]
            cell_embeddings_reshaped = cell_embeddings_arr.reshape(num_cells, SEQ_LEN, 1, 200)
            gene_embeddings = np.sum(cell_embeddings_reshaped * attn_scores_norm[:, :, None, None], axis=0).mean(1)
            
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

        # add noise
        for i in range(len(train_data)):
            if np.random.rand() < self.args.noise:
                train_data[i][2] = 1 - train_data[i][2]

        loader = load_data(data_input)
        feature2 = loader.exp_data()
        feature2 = torch.from_numpy(feature2)
        gene_num = feature2.shape[0]
        feature1 = self._get_embeddings(gene_num, data_input)
        self.input_dim = feature2.size()[1]
        self.gene_dim = feature1.size()[1]

        data_feature2 = feature2.to(self.device)
        data_feature1 = feature1.to(self.device)
        train_load = scRNADataset(train_data, gene_num, flag=self.args.flag)
        adj = train_load.Adj_Generate(tf, loop=self.args.loop)
        adj = adj2saprse_tensor(adj)
        adj = adj.to(self.device)
        train_data = torch.from_numpy(train_data)
        train_data = train_data.to(self.device)
        test_data = torch.from_numpy(test_data)
        test_data = test_data.to(self.device)

        return train_load, test_data, adj, data_feature1, data_feature2


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


    def train(self):
        max_AUC = 0
        accumulate_patience = 0
        train_load, test_data, adj, data_feature1, data_feature2 = self._prepare_data()
        self.model = self.get_model()
        optimizer = getattr(optim, self.args.optimizer_name)(self.model.parameters(), lr=self.args.gnn_lr, weight_decay=self.args.gnn_weight_decay)

        for epoch in tqdm(range(self.args.gnn_epochs)):
            running_loss = 0.0
            for train_x, train_y in DataLoader(train_load, batch_size=self.args.batch_size, shuffle=True):
                self.model.train()
                optimizer.zero_grad()

                if self.args.flag:
                    train_y = train_y.to(self.device)
                else:
                    train_y = train_y.to(self.device).view(-1, 1)

                pred = self.model(data_feature2, adj, train_x, data_feature1)

                if self.args.flag:
                    pred = torch.softmax(pred, dim=1)
                else:
                    pred = torch.sigmoid(pred)

                loss_BCE = F.binary_cross_entropy(pred, train_y)
                loss_BCE.backward()
                optimizer.step()

                running_loss += loss_BCE.item()
            
            if (epoch+1) % self.args.gnn_eval_interval == 0:
                self.model.eval()
                score = self.model(data_feature2, adj, test_data, data_feature1)

                if self.args.flag:
                    score = torch.softmax(score, dim=1)
                else:
                    score = torch.sigmoid(score)

                AUC, AUPR, AUPR_norm = Evaluation(y_pred=score, y_true=test_data[:, -1],flag=self.args.flag)

                if AUC > max_AUC:
                    accumulate_patience = 0
                    max_AUC = AUC
                    AUC_AUPR = AUPR
                    # AUC_epoch = epoch
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(self.args.ckpt_dir, f"model_seed{self.args.random_seed}.pt"),
                    )
                    save_args(self.args, self.args.ckpt_dir)
                else:
                    accumulate_patience += 1
                    if accumulate_patience >= 10:
                        break

        logger.info(f"best_auroc: {max_AUC:.4f}, auprc: {AUC_AUPR:.4f}")
        return max_AUC, AUC_AUPR
    

def main():
    set_logging()
    args = parse_args()
    logger.critical(
        f"Training on {args.dataset} with {args.noise} noise, {args.llm_type} as scFM backbone and {args.gnn_type} as GNN backbone"
    )
    logger.info(args)

    args.random_seed = args.start_seed
    set_seed(random_seed=args.random_seed)

    trainer = Trainer(args)
    AUROC, AUPRC = trainer.train()
    logger.info(AUROC, AUPRC)

    del trainer
    torch.cuda.empty_cache()
    gc.collect()
    return AUROC, AUPRC


if __name__ == "__main__":
    AUROC, AUPRC = main()
    print(AUROC, AUPRC)