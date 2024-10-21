import sys

sys.path.append('../')
from multiprocessing import Process

from omegaconf import OmegaConf
from src.utils.run_lib import *
from src.utils.counting_utils import * 

calib_val_frac = [0.5] 

top_lbl_hb_params     =   { 
                            "points_per_bin": [50,25],
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
                        "training_conf_t.learning_rate": [0.0001],

                        "calib_val_frac": calib_val_frac
                        }

auto_lbl_opt_v1_params = {  "training_conf_g.optimizer" : ["adam"],
                            "training_conf_t.optimizer" : ["adam"],
                            "l1": [0.01,0.05,0.1],
                            "l2": [1.0],
                            "training_conf_g.learning_rate": [0.0001, 0.0005, 0.001], 
                            "training_conf_t.learning_rate": [0.0001],

                            "calib_val_frac": calib_val_frac
                        }


# add training time calib confs 
std_xent_params        =  {   }
squentropy_params      =  {   }

label_smoothing_params =   {  
                            "label_smoothing" : [0.15] 
                            }

focal_params           =  { 
                            "gamma": [2.0]
                        }

crl_params             =  {  
                            "rank_target": ["softmax"], # options : softmax, margin, entropy
                            "rank_weight": [1.0]
                        }

def create_calib_sub_confs():

    lst_calib_confs = [ ] 

    def add_confs(sub_base_conf_fpath,sub_base_conf_params):
        sub_base_conf = OmegaConf.load(sub_base_conf_fpath)
        lst_sub_confs  = create_sub_confs(sub_base_conf, sub_base_conf_params, sub_base_conf['name'] )
        lst_calib_confs.extend(lst_sub_confs)
    



    add_confs(f'{conf_dir}/train-time/xent_calib_conf.yaml',  std_xent_params)

    if(run_post_hoc):
        add_confs(f'{conf_dir}/post-hoc/top_label_hist_bin_base_conf.yaml',        top_lbl_hb_params)

        add_confs(f'{conf_dir}/post-hoc/temp_scaling_base_conf.yaml',              scaling_params)

        add_confs(f'{conf_dir}/post-hoc/scaling_binning_base_conf.yaml',          scaling_binning_params)

        #add_confs(f'{conf_dir}/post-hoc/auto_lbl_opt_v0_calib.yaml',              auto_lbl_opt_v0_params)

        #add_confs(f'{conf_dir}/post-hoc/auto_lbl_opt_v1_calib.yaml',              auto_lbl_opt_v1_params)

    if(run_train_time):

        add_confs(f'{conf_dir}/train-time/squentropy_calib_conf.yaml',              squentropy_params)

        add_confs(f'{conf_dir}/train-time/label_smoothing_calib_conf.yaml',         label_smoothing_params)

        add_confs(f'{conf_dir}/train-time/focal_calib_conf.yaml',                   focal_params)

        add_confs(f'{conf_dir}/train-time/crl_calib_conf.yaml',                     crl_params)
    
    return lst_calib_confs



if __name__ == "__main__":
    
    root_dir = '../'
    conf_dir = f'{root_dir}configs/calib-exp/'

    root_pfx = 'unit_ball_calib-exp-runs2'

    base_conf                = OmegaConf.load('{}/unit_ball_base_conf_torch.yaml'.format(conf_dir))
    base_conf['output_root'] = f'{root_dir}/outputs/{root_pfx}/'


    # compute configs
    lst_devices    = ['cuda:0']

    run_batch_size = 10
    overwrite_flag = False  # False ==> don't overwrite, True ==> overwrite existing results 


    run_post_hoc   = True 
    #run_post_hoc   = False 

    run_train_time = True 
    #run_train_time = False 

    run_confs      = True 
    #run_confs      = False 

    #dump_results   = True 
    dump_results   = False 

    # Root level config parameters

    T                     = 1 # number of random seeds ( tirals)
    lst_C1                = [0.25]
    lst_C                 = [2]
    lst_eps               = [0.05]
    lst_seed_frac         = [0.2]
    lst_query_batch_frac  = [0.05]
    #lst_max_num_train_pts = [250,500,1000]
    lst_max_num_train_pts = [500]
    lst_max_num_val_pts   = [2000]
    lst_methods           = ['active_labeling']


    lst_seeds             = [i for i in range(T)] # Our secrete sauce or let's say chutney :D 

    lst_calib_confs       = create_calib_sub_confs()

    params = {
            'C_1'               : lst_C1, 
            'eps'               : lst_eps,
            'seed'              : lst_seeds,
            'method'            : lst_methods,
            'C'                 : lst_C,
            'calib_conf'        : lst_calib_confs,
            'seed_frac'         : lst_seed_frac,
            'query_batch_frac'  : lst_query_batch_frac,
            'max_num_train_pts' : lst_max_num_train_pts,
            'max_num_val_pts'   : lst_max_num_val_pts
            }


    lst_confs          = create_confs(base_conf,params)

    print(lst_confs)
    print(f'Total Confs to run {len(lst_confs)}')

    if(run_confs):
        lst_confs = list(sorted(lst_confs, key=lambda x: x['method']))

        batched_par_run(lst_confs,batch_size=run_batch_size,
                        lst_devices=lst_devices,overwrite=overwrite_flag)
    
    if(dump_results):
        keys = ['calib_conf','C_1', 'eps','max_num_train_pts','max_num_val_pts','method','query_batch_frac','seed_frac']
        keys+= list(top_lbl_hb_params.keys()) + list(scaling_params.keys()) + list(auto_lbl_opt_v1_params.keys())
        #print(keys)
        save_results(root_pfx,base_conf['output_root'],keys)
