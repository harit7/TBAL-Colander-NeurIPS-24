
#! /bin/bash

data_model_key=cifar10_resnet18

run_id=${1:-"eval_hyp"}  # default 0 # prev 0
T=${2:-3}  # default 3
num_gpu=${3:-1}  # default 1, cuda:0
jobs_per_gpu=${4:-12}    # default 10 

echo "data_model_key":$data_model_key
echo "run_id ":$run_id
echo "T": $T
echo "num_gpu":$num_gpu 
echo "jobs_per_gpu":$jobs_per_gpu

python main_script.py --command run --eval hyp --data_model_key $data_model_key \
                      --method passive_learning  --hyp_train fixed --hyp_post search \
                      --T $T --num_gpu $num_gpu  --jobs_per_gpu $jobs_per_gpu --run_id $run_id
