import logging
import warnings

from optuna.exceptions import ExperimentalWarning

from .HP_search import Single_HP_search

logger = logging.getLogger(__name__)
# optuna.logging.set_verbosity(optuna.logging.ERROR)
warnings.filterwarnings("ignore", category=ExperimentalWarning, module="optuna.multi_objective")
warnings.filterwarnings("ignore", category=FutureWarning)


class GNN_HP_search(Single_HP_search):
    def setup_search_space(self, args, trial):
        args.gnn_lr = trial.suggest_float("gnn_lr", 1e-5, 1e-2, log=True)
        args.gnn_weight_decay = trial.suggest_float("gnn_weight_decay", 1e-7, 1e-4, log=True)
        args.dropout = trial.suggest_float("gnn_dropout", 0.1, 0.8)
        args.gnn_num_layers = trial.suggest_int("gnn_num_layers", 1, 6)
        args.mlp_num_layers = trial.suggest_int("mlp_num_layers", 1, 6)
        args.batch_size = trial.suggest_int("batch_size", 32, 256)
        
        args.gnn_hidden_dims = [
            trial.suggest_int(f"gnn_hidden_dim_l{i}", 4, 256) 
            for i in range(args.gnn_num_layers)
        ]
            
        args.mlp_hidden_dims = [
            trial.suggest_int(f"mlp_hidden_dim_l{i}", 4, 256) 
            for i in range(args.mlp_num_layers)
        ]

        args.optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
        return args
    
    
class GAT_HP_search(Single_HP_search):
    def setup_search_space(self, args, trial):
        args.gnn_lr = trial.suggest_float("gnn_lr", 1e-5, 1e-2, log=True)
        args.gnn_weight_decay = trial.suggest_float("gnn_weight_decay", 1e-7, 1e-4, log=True)
        args.dropout = trial.suggest_float("gnn_dropout", 0.1, 0.8)
        # args.gnn_num_layers = trial.suggest_int("gnn_num_layers", 1, 4) # for mDC
        args.gnn_num_layers = trial.suggest_int("gnn_num_layers", 1, 6)
        args.mlp_num_layers = trial.suggest_int("mlp_num_layers", 1, 6)
        args.batch_size = trial.suggest_int("batch_size", 32, 256)
        
        args.gnn_hidden_dims = [
            trial.suggest_int(f"gnn_hidden_dim_l{i}", 4, 256) 
            for i in range(args.gnn_num_layers)
        ]
            
        args.mlp_hidden_dims = [
            trial.suggest_int(f"mlp_hidden_dim_l{i}", 4, 256) 
            for i in range(args.mlp_num_layers)
        ]

        args.num_heads = [
            trial.suggest_int(f"num_heads_l{i}", 1, 8) 
            for i in range(args.gnn_num_layers)
        ]

        args.reduction = trial.suggest_categorical("reduction", ["concate", "mean"])
        args.alpha = trial.suggest_float("alpha", 0.01, 0.5)
        return args