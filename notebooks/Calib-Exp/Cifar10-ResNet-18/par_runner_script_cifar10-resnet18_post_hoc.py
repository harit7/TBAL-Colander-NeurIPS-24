import sys
sys.path.append('../../../')
sys.path.append('../../')
sys.path.append('../')

from multiprocessing import Process
from omegaconf import OmegaConf
from core.run_lib import *
from utils.counting_utils import * 

root_dir = '../../../'
conf_dir = f'{root_dir}configs/calib-exp/'

root_pfx = 'cifar10-resnet18_post_hoc_calib-exp-runs'

base_conf                = OmegaConf.load('{}/cifar10_resnet18_base_conf.yaml'.format(conf_dir))
base_conf['output_root'] = f'{root_dir}/outputs/{root_pfx}/'

# compute configs
lst_devices    = ['cuda:0']

run_batch_size = 6 
overwrite_flag = False  # False ==> don't overwrite, True ==> overwrite existing results 
#run_confs      = False 
dump_results   = True 

# Root level config parameters

T                     = 1 # number of random seeds ( tirals)
lst_C1                = [0.25]
lst_C                 = [2]
lst_eps               = [0.05]
lst_seed_frac         = [0.2]
lst_query_batch_frac  = [0.05]
#lst_max_num_train_pts = [250,500,1000]
lst_max_num_train_pts = [10000,20000] 
lst_max_num_val_pts   = [10000,20000]

lst_methods           = ['active_labeling']

lst_seeds             = [i for i in range(T)] # Our secrete sauce or let's say chutney :D 


### Create calibration configs 

top_lbl_hb_calib_base_conf       = OmegaConf.load(f'{conf_dir}/post-hoc/top_label_hist_bin_base_conf.yaml')
scaling_calib_base_conf          = OmegaConf.load(f'{conf_dir}/post-hoc/temp_scaling_base_conf.yaml')
scaling_binning_calib_base_conf  = OmegaConf.load(f'{conf_dir}/post-hoc/scaling_binning_base_conf.yaml')

auto_lbl_opt_v0_base_conf        = OmegaConf.load(f'{conf_dir}/post-hoc/auto_lbl_opt_v0_calib.yaml')
auto_lbl_opt_v1_base_conf        = OmegaConf.load(f'{conf_dir}/post-hoc/auto_lbl_opt_v1_calib.yaml')


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


lst_top_lbl_hb_calib_confs       =  create_sub_confs(top_lbl_hb_calib_base_conf,
                                                     top_lbl_hb_params, 
                                                     top_lbl_hb_calib_base_conf['name'] )

lst_scaling_calib_confs          =  create_sub_confs(scaling_calib_base_conf,
                                                     scaling_params, 
                                                     scaling_calib_base_conf['name'] ) 

lst_scaling_binning_calib_confs  =  create_sub_confs(scaling_binning_calib_base_conf,
                                                     scaling_binning_params, 
                                                     scaling_binning_calib_base_conf['name'] ) 

lst_auto_lbl_opt_v0_confs        =  create_sub_confs(auto_lbl_opt_v0_base_conf,
                                                     auto_lbl_opt_v0_params, 
                                                     auto_lbl_opt_v0_base_conf['name'] )  
   
lst_auto_lbl_opt_v1_confs        =  create_sub_confs(auto_lbl_opt_v1_base_conf,
                                                     auto_lbl_opt_v1_params, 
                                                     auto_lbl_opt_v1_base_conf['name'] )  


lst_calib_confs = [] 
lst_calib_confs.extend([None])

lst_calib_confs.extend(lst_top_lbl_hb_calib_confs)

lst_calib_confs.extend(lst_scaling_calib_confs)

#lst_calib_confs.extend(lst_scaling_binning_calib_confs)

lst_calib_confs.extend(lst_auto_lbl_opt_v0_confs)

lst_calib_confs.extend(lst_auto_lbl_opt_v1_confs)


params = {
        'C_1'               : lst_C1, 
        'eps'               : lst_eps,
        'seed'              : lst_seeds,
        'method'            : lst_methods,
        'C'                 : lst_C,
        'seed_frac'         : lst_seed_frac,
        'query_batch_frac'  : lst_query_batch_frac,
        'max_num_train_pts' : lst_max_num_train_pts,
        'max_num_val_pts'   : lst_max_num_val_pts
        }

params_post_hoc = copy.deepcopy(params) 
params_post_hoc['calib_conf']  = lst_calib_confs 


if __name__ == "__main__":
    
    lst_confs          = create_confs(base_conf,params_post_hoc)


    print(f'Total Confs to run {len(lst_confs)}')

    lst_confs = list(sorted(lst_confs, key=lambda x: x['method']))
    
    batched_par_run(lst_confs,batch_size=run_batch_size,lst_devices=lst_devices,overwrite=overwrite_flag)
    
    if(dump_results):
        keys = ['calib_conf','C_1', 'eps','max_num_train_pts','max_num_val_pts','method','query_batch_frac','seed_frac']
        keys+= list(top_lbl_hb_params.keys()) + list(scaling_params.keys()) + list(auto_lbl_opt_v1_params.keys())
        #print(keys)
        save_results(root_pfx,base_conf['output_root'],keys)
