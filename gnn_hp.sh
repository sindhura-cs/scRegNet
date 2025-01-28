dataset=$1
gnn_type=$2
cell_type=$3
num_TF=$4
scFM_type=$5


bash optuna.sh --gnn_type $gnn_type --dataset $dataset  --llm_type $scFM_type \
    --gnn_eval_interval 5 \
    --batch_size 256 \
    --gnn_epochs 50 \
    --n_trials 3 \
    --cell_type $cell_type \
    --num_TF $num_TF \
    --single_gpu 2