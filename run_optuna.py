import logging
import warnings
from optuna.exceptions import ExperimentalWarning
from src.args import parse_args
from src.optuna.search_space import (
    GNN_HP_search,
    GAT_HP_search,
)
from src.utils import set_logging

logger = logging.getLogger(__name__)
# optuna.logging.set_verbosity(optuna.logging.ERROR)
warnings.filterwarnings("ignore", category=ExperimentalWarning, module="optuna.multi_objective")
warnings.filterwarnings("ignore", category=FutureWarning)


def get_search_instance(gnn_type):
    if gnn_type in ["GAT"]:
        return GAT_HP_search
    elif gnn_type in ["GraphSAGE", "GCN"]:
        return GNN_HP_search
    else:
        raise NotImplementedError("not implemented HP search class")


def main():
    set_logging()
    args = parse_args()
    hp_search = get_search_instance(args.gnn_type)(args)
    if args.load_study:
        hp_search.load_study()
    else:
        logger.critical(
            f"Start HP search: Run Optuna with GNN model '{args.gnn_type}' and scFM model '{args.llm_type}' on '{args.dataset}' dataset for {args.n_trials} trials."
        )
        hp_search.run(n_trials=args.n_trials)


if __name__ == "__main__":
    main()
