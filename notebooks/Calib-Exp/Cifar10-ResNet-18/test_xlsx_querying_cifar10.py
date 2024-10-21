import sys
sys.path.append('../../../')
sys.path.append('../../')
sys.path.append('../')

from multiprocessing import Process
from omegaconf import OmegaConf
from core.run_lib import *
from utils.counting_utils import * 

root_pfx = 'cifar10-resnet18_post_hoc_calib-exp-runs'
root_dir = '../../../'
conf_dir = f'{root_dir}configs/calib-exp/'
base_conf                = OmegaConf.load('{}/cifar10_resnet18_base_conf.yaml'.format(conf_dir))
base_conf['output_root'] = f'{root_dir}/outputs/{root_pfx}/'

calib_val_frac = [0.5] 
top_lbl_hb_params     =   { 
                             "points_per_bin": [50],
                              "calib_val_frac": calib_val_frac

                          }

scaling_params         =  {  "training_conf.optimizer" : ['adam'],
                             "training_conf.learning_rate": [0.5,0.1],
                             "calib_val_frac": calib_val_frac 
                          }

scaling_binning_params = {
                            'training_conf.num_bins': [10,15,20],
                             "training_conf.learning_rate": [0.5,0.1], 
                             "calib_val_frac": calib_val_frac
                         }

auto_lbl_opt_v0_params = {
                           "training_conf_g.optimizer" : ["adam"],
                           "training_conf_g.learning_rate": [0.0001, 0.0005, 0.001], 
                           #"training_conf_t.learning_rate": [0.0001, 0.0005, 0.001],
                           "training_conf_t.learning_rate": [0.0001,0.001],

                           "calib_val_frac": calib_val_frac
                         }

auto_lbl_opt_v1_params = {  "training_conf_g.optimizer" : ["adam"],
                            "training_conf_t.optimizer" : ["adam"],
                            "l1": [0.01,0.05,0.1],
                            "l2": [1.0],
                            "training_conf_g.learning_rate": [0.0001, 0.0005, 0.001], 
                            "training_conf_t.learning_rate": [0.0001,0.001],

                            "calib_val_frac": calib_val_frac
                         }

keys = ['calib_conf','C_1', 'eps','max_num_train_pts','max_num_val_pts','method','query_batch_frac','seed_frac']
keys+= list(top_lbl_hb_params.keys()) + list(scaling_params.keys()) + list(auto_lbl_opt_v1_params.keys())
#print(keys)
breakpoint()
save_results(root_pfx,base_conf['output_root'],keys)