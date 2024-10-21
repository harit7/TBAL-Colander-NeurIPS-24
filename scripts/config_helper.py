import os
from omegaconf import OmegaConf
from src.utils.conf_utils import *
import math


### Create calibration configs
def add_confs(sub_base_conf, sub_base_conf_params, lst_sub_confs, extra_keys=[]):
    lst_confs_tmp = create_sub_confs(
        sub_base_conf, sub_base_conf_params, sub_base_conf["name"]
    )
    print(f"Number of sub_confs for {sub_base_conf['name'] } : {len(lst_confs_tmp)}")
    lst_sub_confs.extend(lst_confs_tmp)

    if len(sub_base_conf_params) > 0:
        for kk in sub_base_conf_params.keys():
            if kk not in extra_keys:
                extra_keys.append(kk)

    return lst_confs_tmp


def copy_global_params(globals, locals):
    for k in globals.keys():
        if k not in locals:
            locals[k] = globals[k]


def make_configs(
    args,
    all_params,
    train_time_methods,
    post_hoc_methods,
    base_conf,
    conf_dir,
    should_filter_bigger_Nv=False,
):
    lst_calib_confs = [None]

    #tmp 
    #lst_calib_confs = [] 
    
    lst_train_confs = []
    extra_keys = []
    for method in train_time_methods:
        sub_base_conf = OmegaConf.load(
            os.path.join(conf_dir, "training_confs", f"{method}_conf.yaml")
        )

        local_params = all_params["train_time"][f"{method}_params"]
        global_params = all_params["train_time"]["global_train_params"]
        copy_global_params(global_params, local_params)
        add_confs(sub_base_conf, local_params, lst_train_confs, extra_keys)

    for method in post_hoc_methods:
        sub_base_conf = OmegaConf.load(
            os.path.join(conf_dir, "post-hoc", f"{method}_conf.yaml")
        )

        local_params = all_params["post_hoc"][f"{method}_params"]
        global_params = all_params["post_hoc"]["global_calib_params"]
        copy_global_params(global_params, local_params)
        add_confs(
            sub_base_conf,
            all_params["post_hoc"][f"{method}_params"],
            lst_calib_confs,
            extra_keys,
        )

    lst_seeds = [i for i in range(args.T)]  # Our secrete sauce

    common_params = all_params["common"]
    params = {
        "C_1": common_params["C_1"],
        "eps": common_params["eps"],
        "seed": lst_seeds,
        "method": [args.method],
        "C": common_params["C"],
        "seed_frac": common_params["seed_frac"],
        "query_batch_frac": common_params["query_batch_frac"],
        "max_num_train_pts": common_params["N_t"],
        "max_num_val_pts": common_params["N_v"],
        "num_hyp_val_samples" : common_params["N_hyp_v"],
    }
    m = math.prod([len(params[k]) for k in params.keys()])

    print(f"num global confs : {m} ")
    cross_train_post_hoc = True
    add_train_post_hoc = False

    if cross_train_post_hoc:
        # this will create cross product of training configs and post-hoc calibration methods.
        # will be a lot too run, so be careful in setting values above.
        params_cp = copy.deepcopy(params)

        params2 = {"training_conf": lst_train_confs, "calib_conf": lst_calib_confs}
        params_cp.update(params2)
        print(len(lst_calib_confs))
        lst_confs = create_confs(base_conf, params_cp, should_filter_bigger_Nv)

    elif add_train_post_hoc:
        # this will create configs with training configs + no (default) post-hoc calibration
        # and configs with default training config and all post-hoc configs created above.

        params_cp = copy.deepcopy(params)

        params_cp.update({"training_conf": lst_train_confs, "calib_conf": [None]})
        lst_confs_1 = create_confs(base_conf, params_cp, should_filter_bigger_Nv)

        params_cp = copy.deepcopy(params)

        params_cp.update(
            {"training_conf": std_xent_train_confs, "calib_conf": lst_calib_confs}
        )
        lst_confs_2 = create_confs(base_conf, params_cp, should_filter_bigger_Nv)

        lst_confs = lst_confs_1 + lst_confs_2

    return lst_confs, extra_keys
