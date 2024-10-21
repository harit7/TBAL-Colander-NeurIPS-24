
#! /bin/bash

data_model_key=cifar10_med_net

run_id=${1:-std_xent_g_reuse}  # default 0
T=${2:-5}  # default 3
num_gpu=${2:-2}  # default 1, cuda:0
jobs_per_gpu=${4:-2}    # default 8

echo "data_model_key":$data_model_key
echo "run_id ":$run_id
echo "T": $T
echo "num_gpu":$num_gpu 
echo "jobs_per_gpu":$jobs_per_gpu

python main_script.py --command run --eval full --data_model_key $data_model_key \
                      --method tbal  --hyp_train fixed --hyp_post fixed --train_time_method std_xent \
                      --post_hoc_method  auto_lbl_opt_v0 \
                      --include_nan_auto_err True --T $T --num_gpu $num_gpu  --jobs_per_gpu $jobs_per_gpu --run_id $run_id
