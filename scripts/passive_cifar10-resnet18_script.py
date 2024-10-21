import sys

sys.path.append('../')
from multiprocessing import Process
from omegaconf import OmegaConf
from src.utils.run_lib import *
from src.utils.counting_utils import *
from src.utils.conf_utils import *
import math

root_dir = '../'
conf_dir = f'{root_dir}configs/calib-exp/'

method = 'passive_learning' # SINGLE round auto-labeling

model_ds_key = 'cifar10_resnet18'

root_pfx = f'{model_ds_key}_calib-v2_{method}'

base_conf                = OmegaConf.load(f'{conf_dir}/{model_ds_key}_base_conf.yaml')
base_conf['output_root'] = f'{root_dir}/outputs/{root_pfx}/'
base_conf['root_dir']  = root_dir

# compute configs
lst_devices    = ['cuda:0']

run_batch_size = 3 
overwrite_flag = False  # False ==> don't overwrite, True ==> overwrite existing results

make_confs     = False
run_confs      = True
#run_confs      = False

dump_results   = True
#dump_results   = False

run_post_hoc   = True
#run_post_hoc   = False

run_train_time = True
#run_train_time = False

# Root level config parameters

T                     = 3 # number of random seeds ( tirals)
lst_C1                = [0.25]
lst_C                 = [2]
lst_eps               = [0.05]
lst_seed_frac         = [0.2]
lst_query_batch_frac  = [0.05]
lst_max_num_train_pts = [10000, 20000]
lst_max_num_val_pts   = [10000,20000]

lst_methods            = ['passive_learning']

lst_seeds             = [i for i in range(T)] # Our secrete sauce or let's say chutney :D


### Create calibration configs

top_lbl_hb_calib_base_conf       = OmegaConf.load(f'{conf_dir}/post-hoc/top_label_hist_bin_base_conf.yaml')
scaling_calib_base_conf          = OmegaConf.load(f'{conf_dir}/post-hoc/temp_scaling_base_conf.yaml')
scaling_binning_calib_base_conf  = OmegaConf.load(f'{conf_dir}/post-hoc/scaling_binning_base_conf.yaml')

#auto_lbl_opt_v0_base_conf        = OmegaConf.load(f'{conf_dir}/post-hoc/auto_lbl_opt_v0_calib.yaml')
#auto_lbl_opt_v1_base_conf        = OmegaConf.load(f'{conf_dir}/post-hoc/auto_lbl_opt_v1_calib.yaml')
auto_lbl_opt_v2_base_conf        = OmegaConf.load(f'{conf_dir}/post-hoc/auto_lbl_opt_v2_calib.yaml')
dirichlet_base_conf              = OmegaConf.load(f'{conf_dir}/post-hoc/dirichlet_base_conf.yaml')




### training-time calibration configs load
std_xent_base_conf                   = OmegaConf.load(f'{conf_dir}/train-time/xent_calib_conf.yaml')
squentropy_base_conf                 = OmegaConf.load(f'{conf_dir}/train-time/squentropy_calib_conf.yaml')
label_smoothing_base_conf            = OmegaConf.load(f'{conf_dir}/train-time/label_smoothing_calib_conf.yaml')
focal_base_conf                      = OmegaConf.load(f'{conf_dir}/train-time/focal_calib_conf.yaml')
crl_base_conf                        = OmegaConf.load(f'{conf_dir}/train-time/crl_calib_conf.yaml')
mixup_base_conf                      = OmegaConf.load(f'{conf_dir}/train-time/mixup_calib_conf.yaml')
mmce_base_conf                       = OmegaConf.load(f'{conf_dir}/train-time/mmce_calib_conf.yaml')
fmfp_base_conf                       = OmegaConf.load(f'{conf_dir}/train-time/fmfp_calib_conf.yaml')



calib_val_frac = [0.25, 0.6,0.75]

top_lbl_hb_params     =   {
                             "points_per_bin": [50,25],
                              "calib_val_frac": calib_val_frac

                          }

scaling_params         =  {  "training_conf.optimizer" : ['adam'],
                             "training_conf.learning_rate": [0.5,0.1],
                             "calib_val_frac": calib_val_frac
                          }

scaling_binning_params = {
                            'training_conf.num_bins': [10,20],
                             "training_conf.learning_rate": [0.5,0.1],
                             "calib_val_frac": calib_val_frac
                         }

dirichlet_params      =   {  "calib_val_frac" : calib_val_frac }

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
auto_lbl_opt_v2_params = {
                            "l1": [1.0],
                            "l2": [0.0,1.0],
                            "l3": [1.0, 15.0],
                            "l4": [1.0,2.0],
                            "regularize" : [False],
                            "training_conf_g.optimizer" : ["adam"],
                            "training_conf_g.learning_rate": [0.0001],
                            "training_conf_g.batch_size": [256,512,1024],
                            "training_conf_g.max_epochs": [500],
                            "features_key" : [ "pre_logits"],
                            "calib_val_frac": calib_val_frac,

                         }


# add training time calib confs
std_xent_params        =  {   }
squentropy_params      =  {   }

label_smoothing_params =   { "label_smoothing" : [0.15] }

focal_params           =  {  "gamma": [2.0] }

crl_params             =  {
                             "rank_target": ["softmax"], # options : softmax, margin, entropy
                             "rank_weight": [1.0]
                          }

mixup_params          = { "mixup_alpha": [0.9] }

mmce_params           = { "mmce_coeff" : [0.055] }

fmfp_params           = {}



lst_calib_confs = [ ]
extra_keys = []
def add_confs(sub_base_conf,sub_base_conf_params):
    lst_sub_confs  = create_sub_confs(sub_base_conf, sub_base_conf_params, sub_base_conf['name'] )
    print(f"Number of sub_confs for {sub_base_conf['name'] } : {len(lst_sub_confs)}")
    lst_calib_confs.extend(lst_sub_confs)

    if(len(sub_base_conf_params)>0):
        extra_keys.extend(sub_base_conf_params.keys())


add_confs(std_xent_base_conf,  std_xent_params)


if(run_post_hoc):
    add_confs(top_lbl_hb_calib_base_conf,        top_lbl_hb_params)

    add_confs(scaling_calib_base_conf,           scaling_params)

    add_confs(scaling_binning_calib_base_conf,   scaling_binning_params)

    #add_confs(auto_lbl_opt_v0_base_conf,         auto_lbl_opt_v0_params)

    #add_confs(auto_lbl_opt_v1_base_conf,         auto_lbl_opt_v1_params)
    add_confs(auto_lbl_opt_v2_base_conf,         auto_lbl_opt_v2_params)

    add_confs(dirichlet_base_conf,               dirichlet_params)

if(run_train_time):

    add_confs(squentropy_base_conf,              squentropy_params)

    add_confs(label_smoothing_base_conf,         label_smoothing_params)

    add_confs(focal_base_conf,                   focal_params)

    add_confs(crl_base_conf,                     crl_params)

    add_confs(mixup_base_conf,                    mixup_params)

    add_confs(mmce_base_conf,                     mmce_params)

    add_confs(fmfp_base_conf,                     fmfp_params)



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


if __name__ == "__main__":

    lst_confs = []

    if(len(sys.argv)>1):
        mode = sys.argv[1]

        if(mode=='run'):
            make_confs = True
            run_confs  = True
        elif(mode=='save'):
            make_confs = False
            run_confs  = False
            dump_results = True
        else:
            print('Specify mode: run | save')
            exit()

    if(make_confs or run_confs):
        lst_confs          = create_confs(base_conf,params)
        #print(lst_confs)
        print(f'Total Confs to run {len(lst_confs)}')

    if(run_confs):
        lst_confs = list(sorted(lst_confs, key=lambda x: x['calib_conf']['name']))

        batched_par_run(lst_confs,batch_size=run_batch_size,
                        lst_devices=lst_devices,overwrite=overwrite_flag)

    if(dump_results):
        keys = ['calib_conf','C_1', 'eps','max_num_train_pts','max_num_val_pts','method','query_batch_frac','seed_frac'] + extra_keys
        #keys+= list(top_lbl_hb_params.keys()) + list(scaling_params.keys()) + list(auto_lbl_opt_v2_params.keys())
        print(keys)
        save_results(root_pfx,base_conf['output_root'],keys)
