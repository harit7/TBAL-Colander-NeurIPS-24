root_dir = "../"

import sys

import argparse

sys.path.append(root_dir)

from multiprocessing import Process
from omegaconf import OmegaConf
from src.utils.run_lib import *
from src.utils.counting_utils import *
from src.utils.conf_utils import *
from src.utils.common_utils import *

from config_helper import *

# use this to debug numpy warnings
#import numpy as np 
#np.seterr('raise')

import warnings

warnings.filterwarnings(action='ignore', message='Mean of empty slice')

import torch
torch.set_float32_matmul_precision('high')

parser = argparse.ArgumentParser(description="Options to run script")

parser.add_argument("--run_id", dest="run_id", default="0", type=str)

parser.add_argument(
    "--hyp_common",
    dest="hyp_common",
    default="fixed",
    help="search or fixed",
    choices=["search", "fixed"],
)

parser.add_argument(
    "--hyp_train",
    dest="hyp_train",
    default="search",
    help="search or use fixed hyper params for train time methods.",
    choices=["search", "fixed"],
)

parser.add_argument(
    "--hyp_post",
    dest="hyp_post",
    default=None,
    help="search or use fixed hyper params for post-hoc methods.",
    choices=["search", "fixed", "none"],
)

parser.add_argument(
    "--data_model_key",
    dest="data_model_key",
    default="mnist_lenet",
    help="dataset_model name key to identify the configs etc.",
    choices=[
        "mnist_lenet",
        "cifar10_resnet18",
        "cifar10_med_net",
        "tiny_imagenet_CLIP",
        "twenty_newsgroups",
        "cifar10_vit_small",
        "circles_linear",
    ],
)

parser.add_argument(
    "--method",
    dest="method",
    default="passive_learning",
    help="passive_learning | tbal (multi round)",
    choices=["passive_learning", "tbal", "al_st", "al_st_all"],
)

parser.add_argument(
    "--train_time_method",
    dest="train_time_method",
    default="all",
    help="std_xent | fmfp | crl | squentropy | all",
    choices=["std_xent", "fmfp", "crl", "squentropy", "all"],
)


parser.add_argument(
    "--post_hoc_method",
    dest="post_hoc_method",
    default="all",
    help="auto_lbl_opt_v0 | scaling | dirichlet | scaling_binning | top_lbl_hb | all",
    choices=["auto_lbl_opt_v0", "scaling", "dirichlet", "scaling_binning", "top_lbl_hb"],
)

parser.add_argument(
    "--num_gpu",
    dest="num_gpu",
    default=1,
    type=int,
    help="number of gpus available to run jobs.",
)

parser.add_argument(
    "--gpu_id",
    dest="gpu_id",
    default=None,
    type=str,
    help="chose specific gpu to run program.",
)

parser.add_argument(
    "--jobs_per_gpu",
    dest="jobs_per_gpu",
    default=1,
    type=int,
    help="number of jobs to run in parallel on each gpu",
)

parser.add_argument(
    "--command",
    dest="command",
    default="make_conf",
    help=" make_conf | run | run_ow | save",
    choices=["make_conf", "run", "run_ow", "save"],
)  # choice.

parser.add_argument(
    "--include_nan_auto_err",
    dest="include_nan_auto_err",
    default=False ,
    help="True | False, whether to include runs with 0 coverage and nan error.",
    choices=[True,False],
    type=bool

)
parser.add_argument(
    "--eval", dest="eval", default="full", help="hyp | full", choices=["hyp", "full"]
)
# evaluate on full data or only on the hyp data ( data reserved for hyp search)

parser.add_argument("--T", dest="T", default=3, type=int)

parser.add_argument(
    "--should_filter_bigger_Nv",
    dest="should_filter_bigger_Nv",
    default=0,
    type=int,
)

parser.add_argument(
    "--should_compile",
    dest="should_compile",
    default=False,
    type=bool,
    choices=[True, False]
)

args = parser.parse_args()

# model_ds_key = 'cifar10_resnet18'
# model_ds_key = 'tiny_imagenet_CLIP'
# model_ds_key = 'twenty_newsgroups'

# Define root prefix (directory name) --- '/output/root_pfx' --- to dump output files
root_pfx = f"{args.data_model_key}_calib_{args.run_id}_{args.method}_eval_{args.eval}"

# Get structured metadata from Base config (yaml) file --- (B config)
conf_dir = os.path.join(root_dir, "configs", "calib-exp")
base_conf = OmegaConf.load(
    os.path.join(conf_dir, f"{args.data_model_key}_base_conf.yaml")
)

# Set new metadata in B config
base_conf["output_root"] = os.path.join(root_dir, "outputs", root_pfx)
base_conf["root_dir"] = root_dir
base_conf["root_pfx"] = root_pfx

# base_conf['mode'] = args.mode

base_conf["eval"] = args.eval  #'full'  # or 'hyp'

# 1. search hyper-parameters 
     # eval is hyp, hyp_train search 
     # eval is hyp, fix train time parameters, hyp_post search 

# 2. when we are done with hyp search and found the best hyper parameters for all
# COMBINATION of post-hoc and train-time methods.
# if eval is full, and hyp_post fixed, hyp_train fixed
# hyp_post fixed file name will be 
# ds_model_key+_+ post_fixed_ train_time_method_name

if(args.train_time_method == 'all'):
    train_time_methods = ["std_xent", "crl", "fmfp", "squentropy"]
else:
    train_time_methods = [args.train_time_method]

if(args.post_hoc_method == "all"):
    post_hoc_methods = [
    "auto_lbl_opt_v0",
    "scaling",
    "dirichlet",
    "scaling_binning",
    "top_lbl_hb"]
else: 
    post_hoc_methods = [args.post_hoc_method]



all_params = {}

common_params = read_json_file(
    os.path.join(
        conf_dir,
        "hyp-search",
        args.method,
        args.data_model_key,
        f"{args.data_model_key}_common_{args.hyp_common}.json",
    )
)
train_params = read_json_file(
    os.path.join(
        conf_dir,
        "hyp-search",
        args.method,
        args.data_model_key,
        f"{args.data_model_key}_train_{args.hyp_train}.json",
    )
)

print(train_params)
all_params["common"] = common_params["common"]
all_params["train_time"] = train_params["train_time"]

if args.hyp_post == "fixed":

    post_params = read_json_file(
        os.path.join(
            conf_dir,
            "hyp-search",
            args.method,
            args.data_model_key,
            f"{args.data_model_key}_post_{args.hyp_post}_{args.train_time_method}.json",
        )
    )
    all_params["post_hoc"] = post_params["post_hoc"]

elif args.hyp_post == "search":
    post_params = read_json_file(
        os.path.join(
            conf_dir,
            "hyp-search",
            args.method,
            args.data_model_key,
            f"{args.data_model_key}_post_{args.hyp_post}.json",
        )
    )
    all_params["post_hoc"] = post_params["post_hoc"]
    
else:

    all_params["post_hoc"] = {}
    post_hoc_methods = []


if __name__ == "__main__":
    is_make_confs = False
    is_run_confs = False
    is_overwrite_flag = False
    is_dump_results = False
    is_extra_keys = []

    if args.command == "make_conf":
        is_make_confs = True
        is_run_confs = False
        is_overwrite_flag = False
        is_dump_results = False

    elif args.command == "run_ow":
        is_make_confs = True
        is_run_confs = True
        is_overwrite_flag = True
        is_dump_results = True

    elif args.command == "run":
        is_make_confs = True
        is_run_confs = True
        is_overwrite_flag = False
        is_dump_results = True

    elif args.command == "save":
        is_make_confs = True
        is_run_confs = False
        is_overwrite_flag = False
        is_dump_results = True
    else:
        print("Specify command: make_conf | run_ow | run | save")
        exit()

    # base_conf['model_conf']['should_compile'] = args.should_compile

    if is_make_confs or is_run_confs:
        lst_confs, extra_keys = make_configs(
            args,
            all_params,
            train_time_methods,
            post_hoc_methods,
            base_conf,
            conf_dir,
            should_filter_bigger_Nv=args.should_filter_bigger_Nv,
        )
        print(f"Total Confs to run {len(lst_confs)}")

    if is_run_confs:
        # run_conf(lst_confs[1]) # Single run test
        # compute configs
        run_batch_size = max(1, args.num_gpu) * args.jobs_per_gpu
        lst_devices = (
            [f"cuda:{i}" for i in range(args.num_gpu)] if args.num_gpu > 0 else ["cpu"]
        )

        run_seq = apply_conf_intel(lst_confs,method=args.method, _eval= args.eval)
        print(len(run_seq), [len(run_seq_x) for run_seq_x in run_seq])

        for lst_confs_shard in run_seq:
            batched_par_run(
                lst_confs_shard,
                batch_size=run_batch_size,
                lst_devices=lst_devices,
                overwrite=is_overwrite_flag,
            )

    if is_dump_results:
        keys = [
            "calib_conf",
            "training_conf",
            "C_1",
            "eps",
            "max_num_train_pts",
            "max_num_val_pts",
            "num_hyp_val_samples",
            "method",
            "query_batch_frac",
            "seed_frac",
        ] + extra_keys
        # keys+= list(top_lbl_hb_params.keys()) + list(scaling_params.keys()) + list(auto_lbl_opt_v2_params.keys())
        save_results(root_pfx, base_conf["output_root"], keys, args.include_nan_auto_err)
