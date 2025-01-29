import argparse
import json
import logging
import os

logger = logging.getLogger(__name__)

GNN_LIST = ["GraphSAGE", "GCN", "GAT"]

def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument("--single_gpu", type=int, default=0)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--start_seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--suffix", type=str, default="main")
    parser.add_argument("--deepspeed", type=str, default=None)

    # parameters for data and model storage
    parser.add_argument("--data_folder", type=str, default="./data")
    parser.add_argument("--scFM_folder", type=str, default="./scFM")
    parser.add_argument("--task_type", type=str, default="link_pred")
    parser.add_argument("--output_dir", type=str)  # output dir
    parser.add_argument("--ckpt_dir", type=str)  # ckpt path to save
    parser.add_argument(
        "--ckpt_name", type=str, default="model.pt"
    )  # ckpt name to be loaded
   

    # training hyperparameters
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--accum_interval", type=int, default=1)
    parser.add_argument("--attention_dropout_prob", type=float, default=0.1)
    parser.add_argument("--label_smoothing", type=float, default=0.3)
    parser.add_argument("--warmup_ratio", type=float, default=0.6)
    parser.add_argument("--num_iterations", type=int, default=4)
    parser.add_argument("--optimizer_name", type=str, default="Adam")
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        choices=["linear", "constant"],
    )

    # module hyperparameters
    # gnn parameters
    parser.add_argument("--gnn_epochs", type=int, default=300)
    parser.add_argument("--gnn_eval_interval", type=int, default=5)
    parser.add_argument("--gnn_label_smoothing", type=float, default=0.1)
    parser.add_argument("--gnn_warmup_ratio", type=float, default=0.25)
    parser.add_argument("--gnn_num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--gnn_dim_hidden", type=int, default=256)
    parser.add_argument("--gnn_lr", type=float, default=5e-4)
    parser.add_argument("--gnn_weight_decay", type=float, default=1e-5)
    parser.add_argument(
        "--gnn_lr_scheduler_type",
        type=str,
        default="constant",
        choices=["constant", "linear"],
    )

    # optuna hyperparameters
    parser.add_argument("--expected_valid_acc", type=float, default=0.6)
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--load_study", action="store_true", default=False)

    # grn parameters
    parser.add_argument("--dataset", type=str, default="tf_500")
    parser.add_argument("--gnn_type", type=str, default="GCN")
    parser.add_argument("--llm_type", type=str, default="Geneformer")
    parser.add_argument("--cell_type", type=str, default="hESC")
    parser.add_argument("--num_TF", type=str, default="500")
    parser.add_argument("--flag", type=bool, default=True) 
    parser.add_argument("--loop", type=bool, default=False)
    parser.add_argument("--type", type=str, default="MLP") # score metric
    parser.add_argument("--mlp_dim_hidden", type=int, default=64) 
    parser.add_argument("--mlp_num_layers", type=int, default=2)

    #params for GAT
    parser.add_argument("--reduction", type=str, default="concate")
    parser.add_argument("--num_heads", type=int, default=3)
    parser.add_argument("--alpha", type=int, default=0.2)

    args = parser.parse_args()
    return args


def save_args(args, dir):
    # if int(os.getenv("RANK", -1)) <= 0:
    FILE_NAME = "args.json"
    with open(os.path.join(dir, FILE_NAME), "w") as f:
        json.dump(args.__dict__, f, indent=2)
    logger.info("args saved to {}".format(os.path.join(dir, FILE_NAME)))


def load_args(dir):
    with open(os.path.join(dir, "args.json"), "r") as f:
        args = argparse.Namespace(**json.load(f))
    return args


if __name__ == "__main__":
    args = parse_args()
