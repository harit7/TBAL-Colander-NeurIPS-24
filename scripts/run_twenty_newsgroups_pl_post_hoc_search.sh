
#! /bin/bash

data_model_key=twenty_newsgroups

run_id=${1:-0}  # default 0
T=${2:-3}  # default 3
num_gpu=${3:-1}  # default 1, cuda:0
jobs_per_gpu=${4:-2}    # default 2

echo "data_model_key":$data_model_key
echo "run_id ":$run_id
echo "T": $T
echo "num_gpu":$num_gpu 
echo "jobs_per_gpu":$jobs_per_gpu

python main_script.py --command run --eval full --data_model_key $data_model_key \
                      --method passive_learning  --hyp_train fixed --hyp_post search \
                      --T $T --num_gpu $num_gpu  --jobs_per_gpu $jobs_per_gpu --run_id $run_id
