import sys
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

sys.path.append('../../../')
sys.path.append('../../')
sys.path.append('../')

from multiprocessing import Process
from omegaconf import OmegaConf
from core.run_lib import *

root_dir = '../../../'
conf_dir = f'{root_dir}configs/calib-exp/'

root_pfx = 'cifar10_sn_post_hoc_calib-exp-runs'

root_pfx = f'{root_dir}/outputs/{root_pfx}/'

base_conf                = OmegaConf.load('{}/cifar10_sn_base_conf_torch.yaml'.format(conf_dir))
base_conf['root_pfx']    = root_pfx

# compute configs
lst_devices    = ['cuda:0']

run_batch_size = 10
overwrite_flag = False  # False ==> don't overwrite, True ==> overwrite existing results 

# Root level config parameters

T                     = 1 # number of random seeds ( tirals)
lst_C1                = [0]
lst_C                 = [2]
lst_eps               = [0.01, 0.05]
lst_seed_frac         = [0.2]
lst_query_batch_frac  = [0.05]
lst_max_num_train_pts = [500]
lst_max_num_val_pts   = [1000]

lst_methods           = ['active_labeling']

lst_seeds             = [i for i in range(T)] # Our secrete sauce or let's say chutney :D 


### Create calibration configs 

top_lbl_hb_calib_base_conf    = OmegaConf.load(f'{conf_dir}/post-hoc/top_label_hist_bin_base_conf.yaml')
platt_scaling_calib_base_conf = OmegaConf.load(f'{conf_dir}/post-hoc/platt_scaling_base_conf.yaml')
auto_lbl_opt_base_conf        = OmegaConf.load(f'{conf_dir}/post-hoc/auto_lbl_opt_calib.yaml')
top_lbl_hb_params = { 
                        "points_per_bin": [50]
                     }
scaling_params    =  { "training_conf.optimizer" : ['adam'],
                       "training_conf.learning_rate": [0.5] 
                      }

auto_lbl_opt_params = { "training_conf_g.learning_rate": [0.07], 
                        "training_conf_t.learning_rate": [0.001],
                       }

lst_top_lbl_hb_calib_confs =  create_sub_confs(top_lbl_hb_calib_base_conf,
                                               top_lbl_hb_params, top_lbl_hb_calib_base_conf['name'] )

lst_platt_calib_confs      =  create_sub_confs(platt_scaling_calib_base_conf,
                                               scaling_params, platt_scaling_calib_base_conf['name'] ) 

lst_auto_lbl_opt_confs     =  create_sub_confs(auto_lbl_opt_base_conf,
                                               auto_lbl_opt_params, auto_lbl_opt_base_conf['name'] )     

lst_calib_confs = [] 
lst_calib_confs.extend([None])
# lst_calib_confs.extend(lst_top_lbl_hb_calib_confs)
# lst_calib_confs.extend(lst_platt_calib_confs)
# lst_calib_confs.extend(lst_auto_lbl_opt_confs)

params = {
        'C_1'               : lst_C1, 
        'eps'               : lst_eps,
        'seed'              : lst_seeds,
        'method'            : lst_methods,
        'calib_conf'        : lst_calib_confs,
        'C'                 : lst_C,
        'seed_frac'         : lst_seed_frac,
        'query_batch_frac'  : lst_query_batch_frac,
        'max_num_train_pts' : lst_max_num_train_pts,
        'max_num_val_pts'   : lst_max_num_val_pts}

# Additional hyperparameters to base config
base_conf['data_conf']['num_classes'] = 10
base_conf['train_pts_query_conf']['query_batch_size'] = 50
base_conf['train_pts_query_conf']['query_strategy_name'] = 'margin_random_v2'
base_conf['train_pts_query_conf']['margin_random_v2_constant'] = 2


if __name__ == "__main__":
    
    lst_confs = []
    lst          = create_confs(base_conf,params)
    lst_confs.extend(lst)
    #print(lst_confs[0])
    print(f'Total Confs to run {len(lst_confs)}')

    lst_confs = list(sorted(lst_confs, key=lambda x: x['method']))

    # batched_par_run(lst_confs,batch_size=run_batch_size,lst_devices=lst_devices,overwrite=overwrite_flag)
    run_conf(lst_confs[0])