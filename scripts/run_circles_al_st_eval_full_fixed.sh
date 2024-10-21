
#! /bin/bash

data_model_key=circles_linear

run_id=${1:-"eval_full"}  # default 0 # prev 0
T=${2:-5}  # default 3
num_gpu=${3:-2}  # default 1, cuda:0
jobs_per_gpu=${4:-10}    # default 10 

echo "data_model_key":$data_model_key
echo "run_id ":$run_id
echo "T": $T
echo "num_gpu":$num_gpu 
echo "jobs_per_gpu":$jobs_per_gpu

python main_script.py --command run --eval full --data_model_key $data_model_key \
                      --method al_st --train_time_method std_xent --hyp_train fixed --hyp_post none\
                      --T $T --num_gpu $num_gpu  --jobs_per_gpu $jobs_per_gpu --run_id $run_id
