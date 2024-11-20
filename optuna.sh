gnn_type=$2
dataset=$4
scFM_type=$6

project_dir='.'
output_dir=${project_dir}/out/${gnn_type}/${scFM_type}/${dataset}/
ckpt_dir=${output_dir}/ckpt

mkdir -p ${output_dir}
mkdir -p ${ckpt_dir}

python run_optuna.py \
    --output_dir $output_dir --ckpt_dir $ckpt_dir \
    $@ 2>&1 | tee ${output_dir}/log.txt