root_dir = '../'

import sys
sys.path.append(root_dir)

from multiprocessing import Process
from omegaconf import OmegaConf
from src.utils.run_lib import *
from src.utils.counting_utils import *
from src.utils.conf_utils import *
import math

model_ds_key = 'cifar10_resnet18'
#model_ds_key = 'tiny_imagenet_CLIP'
#model_ds_key = 'twenty_newsgroups'

method = "passive_learning"

root_pfx = f'{model_ds_key}_calib-v3_{method}'

conf_dir =  os.path.join(root_dir , "configs", "calib-exp" )
base_conf                = OmegaConf.load(os.path.join( conf_dir, f"{model_ds_key}_base_conf.yaml"))

base_conf['output_root'] =   os.path.join(root_dir, "outputs", root_pfx )
base_conf['root_dir']    = root_dir
base_conf['root_pfx']    = root_pfx

# compute configs
lst_devices    = ['cuda:0']
#lst_devices    = ['cuda:0']

run_batch_size = 9 
overwrite_flag = False  # False ==> don't overwrite, True ==> overwrite existing results


run_post_hoc   = True
#run_post_hoc   = False

run_train_time = True
#run_train_time = False

run_confs      = False
#run_confs      = False

#dump_results   = True
dump_results   = False

cross_train_post_hoc = False
add_train_post_hoc   = True

# Root level config parameters
T                     = 3 # number of random seeds ( tirals)
lst_C1                = [0.25]
lst_C                 = [2]
lst_eps               = [0.05]
lst_seed_frac         = [0.2]
lst_query_batch_frac  = [0.05]
lst_max_num_train_pts = [10000, 20000]
lst_max_num_val_pts   = [10000,20000]

lst_methods           = [method]
#lst_methods            = ['passive_learning']
#lst_methods           = ['active_labeling', 'passive_learning']

lst_seeds             = [i for i in range(T)] # Our secrete sauce or let's say chutney :D


global_train_params = {
                   "optimizer":      ["sgd"],
                   "learning_rate" : [0.01],
                   "batch_size":     [32],
                   "max_epochs":     [50],
                   "weight_decay":   [1e-4],
                   "momentum":       [0.9]
                  }


global_calib_params = { "calib_val_frac" : [0.3, 0.5]}



# add training time calib confs
std_xent_params        =  {
                                "optimizer": global_train_params['optimizer'],
                                "learning_rate": global_train_params['learning_rate'],
                                "max_epochs": global_train_params['max_epochs'],
                                "batch_size": global_train_params['batch_size'],
                                "weight_decay": global_train_params['weight_decay'],
                                "momentum":  global_train_params['momentum']
                           }

squentropy_params      =   {
                                "optimizer": global_train_params['optimizer'],
                                "learning_rate": global_train_params['learning_rate'],
                                "max_epochs": global_train_params['max_epochs'],
                                "batch_size": global_train_params['batch_size'],
                                "weight_decay": global_train_params['weight_decay'],
                                "momentum":   global_train_params['momentum']
                           }

label_smoothing_params =   {
                               "label_smoothing" : [0.15],
                                "optimizer": global_train_params['optimizer'],
                                "learning_rate": global_train_params['learning_rate'],
                                "max_epochs": global_train_params['max_epochs'],
                                "batch_size": global_train_params['batch_size'],
                                "weight_decay": global_train_params['weight_decay'],
                                "momentum":   global_train_params['momentum']

                            }

focal_params           =  {
                                "gamma": [2.0],
                                "optimizer":global_train_params['optimizer'],
                                "learning_rate": global_train_params['learning_rate'],
                                "max_epochs": global_train_params['max_epochs'],
                                "batch_size": global_train_params['batch_size'],
                                "weight_decay": global_train_params['weight_decay'],
                                "momentum":   global_train_params['momentum']
                            }

crl_params             =  {
                             "rank_target": ["softmax"], # options : softmax, margin, entropy
                             "rank_weight": [1.0],
                             "optimizer":global_train_params['optimizer'],
                             "learning_rate": global_train_params['learning_rate'],
                             "max_epochs": global_train_params['max_epochs'],
                             "batch_size": global_train_params['batch_size'],
                             "weight_decay": global_train_params['weight_decay'],
                             "momentum":   global_train_params['momentum']
                          }

mixup_params          = {
                            "mixup_alpha": [0.9],
                            "optimizer":global_train_params['optimizer'],
                            "learning_rate": global_train_params['learning_rate'],
                            "max_epochs": global_train_params['max_epochs'],
                            "batch_size": global_train_params['batch_size'],
                            "weight_decay": global_train_params['weight_decay'],
                            "momentum":   global_train_params['momentum']
                        }

mmce_params           = {
                            "mmce_coeff" : [0.055],
                            "optimizer":global_train_params['optimizer'],
                            "learning_rate": global_train_params['learning_rate'],
                            "max_epochs": global_train_params['max_epochs'],
                            "batch_size": global_train_params['batch_size'],
                            "weight_decay": global_train_params['weight_decay'],
                            "momentum":   global_train_params['momentum']
                        }

fmfp_params           = {
                            "optimizer": ["sam"], # don't change optimizer for this.
                            "learning_rate": global_train_params['learning_rate'],
                            "max_epochs": global_train_params['max_epochs'],
                            "batch_size": global_train_params['batch_size'],
                            "weight_decay": global_train_params['weight_decay'],
                            "momentum":   global_train_params['momentum']
                        }

#---------------

top_lbl_hb_params     =   {
                             "points_per_bin": [50,25],
                             "calib_val_frac": global_calib_params["calib_val_frac"]

                          }

scaling_params         =  {   "training_conf.optimizer" : ['adam'],
                              "training_conf.learning_rate": [0.5, 0.1],
                              "training_conf.batch_size" : [64],
                              "training_conf.max_epochs": [20],
                              "training_conf.weight_decay": [1.0, 0.1, 0.01],
                              "calib_val_frac": global_calib_params["calib_val_frac"]
                          }

scaling_binning_params = {
                            'training_conf.num_bins': [10,20],
                             "training_conf.learning_rate": [0.5, 0.1],
                             "training_conf.batch_size" : [64],
                             "training_conf.max_epochs": [20],
                             "training_conf.weight_decay": [1.0, 0.1, 0.01],
                             "calib_val_frac": global_calib_params["calib_val_frac"]

                         }

dirichlet_params      =   {
                                "training_conf.optimizer" : ['adam'],
                                "training_conf.learning_rate" : [0.5],
                                "training_conf.reg" : [1e-2],
                                "training_conf.batch_size" : [64],
                                "training_conf.max_epochs": [20],
                                "calib_val_frac": global_calib_params["calib_val_frac"]
                          }


auto_lbl_opt_v0_params = {
                            "l1" : [1.0],
                            "l2":[5.0,10.0],
                            "l3" : [0.0],
                            "features_key" : ["logits","pre_logits"],
                            "class_wise" : ["independent"],
                           "training_conf_g.optimizer" : ["adam"],
                           "training_conf_g.learning_rate": [0.0001, 0.0005, 0.001],
                           "training_conf_g.max_epochs": [1000,2000],
                           "training_conf_g.weight_decay": [1.0, 0.1, 0.01],
                           "training_conf_g.batch_size": [32, 64],
                           "regularize": [True],
                           "alpha_1" : [1.0],
                           "model_conf":["linear", "two_layer : 1 : relu "],

                           #"training_conf_t.learning_rate": [0.0001, 0.0005, 0.001],
                           #"training_conf_t.learning_rate": [0.0001],

                           "calib_val_frac": global_calib_params["calib_val_frac"]
                         }


### Create calibration configs

top_lbl_hb_calib_base_conf       = OmegaConf.load( os.path.join( conf_dir, "post-hoc", "top_label_hist_bin_base_conf.yaml"))
scaling_calib_base_conf          = OmegaConf.load( os.path.join( conf_dir, "post-hoc", "temp_scaling_base_conf.yaml"))
scaling_binning_calib_base_conf  = OmegaConf.load( os.path.join( conf_dir, "post-hoc","scaling_binning_base_conf.yaml"))

auto_lbl_opt_v0_base_conf        = OmegaConf.load(os.path.join( conf_dir, "post-hoc",'auto_lbl_opt_v0_calib.yaml'))
auto_lbl_opt_v1_base_conf        = OmegaConf.load(os.path.join( conf_dir, "post-hoc", 'auto_lbl_opt_v1_calib.yaml'))
auto_lbl_opt_v2_base_conf        = OmegaConf.load(os.path.join( conf_dir, "post-hoc", "auto_lbl_opt_v2_calib.yaml"))
dirichlet_base_conf              = OmegaConf.load(os.path.join( conf_dir, "post-hoc", "dirichlet_base_conf.yaml"))

### training-time calibration configs load
std_xent_base_conf                   = OmegaConf.load( os.path.join(conf_dir, "training_confs","xent_conf.yaml"))
squentropy_base_conf                 = OmegaConf.load( os.path.join(conf_dir, "training_confs", "squentropy_conf.yaml"))
label_smoothing_base_conf            = OmegaConf.load( os.path.join(conf_dir, "training_confs", "label_smoothing_conf.yaml"))
focal_base_conf                      = OmegaConf.load( os.path.join(conf_dir, "training_confs", "focal_conf.yaml"))
crl_base_conf                        = OmegaConf.load( os.path.join(conf_dir, "training_confs", "crl_conf.yaml"))
mixup_base_conf                      = OmegaConf.load( os.path.join(conf_dir, "training_confs", "mixup_conf.yaml"))
mmce_base_conf                       = OmegaConf.load( os.path.join(conf_dir, "training_confs", "mmce_conf.yaml"))
fmfp_base_conf                       = OmegaConf.load( os.path.join(conf_dir, "training_confs", "fmfp_conf.yaml"))


lst_calib_confs = [ None ]

lst_train_confs = [ ]

# for each base_training_conf in lst_train_confs create loss function specific config by overwriting given params

extra_keys = []

def add_confs(sub_base_conf, sub_base_conf_params, lst_sub_confs ):

    lst_confs_tmp  = create_sub_confs(sub_base_conf, sub_base_conf_params, sub_base_conf['name'] )
    print(f"Number of sub_confs for {sub_base_conf['name'] } : {len(lst_confs_tmp)}")
    lst_sub_confs.extend(lst_confs_tmp)

    if(len(sub_base_conf_params)>0):
        for kk in sub_base_conf_params.keys():
            if(kk  not in extra_keys):
                extra_keys.append(kk)

    return lst_confs_tmp

std_xent_train_confs = add_confs(  std_xent_base_conf,  std_xent_params , lst_train_confs)

if(run_train_time):

    add_confs( squentropy_base_conf, squentropy_params, lst_train_confs )

    add_confs( label_smoothing_base_conf, label_smoothing_params, lst_train_confs)

    add_confs( focal_base_conf, focal_params, lst_train_confs)

    add_confs( crl_base_conf, crl_params, lst_train_confs)

    add_confs( mixup_base_conf, mixup_params, lst_train_confs)

    add_confs( mmce_base_conf, mmce_params, lst_train_confs)

    add_confs( fmfp_base_conf, fmfp_params, lst_train_confs)


if(run_post_hoc):
    add_confs(top_lbl_hb_calib_base_conf, top_lbl_hb_params, lst_calib_confs)

    add_confs( scaling_calib_base_conf, scaling_params , lst_calib_confs)

    add_confs(scaling_binning_calib_base_conf,   scaling_binning_params, lst_calib_confs)

    add_confs(auto_lbl_opt_v0_base_conf,  auto_lbl_opt_v0_params, lst_calib_confs)

    #add_confs(auto_lbl_opt_v1_base_conf, auto_lbl_opt_v1_params, lst_calib_confs)
    #add_confs( auto_lbl_opt_v2_base_conf, auto_lbl_opt_v2_params, lst_calib_confs)

    add_confs( dirichlet_base_conf, dirichlet_params, lst_calib_confs)


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



if __name__ == "__main__":

    if(len(sys.argv)>1):
        mode = sys.argv[1]

        make_confs = False
        run_confs  = False
        overwrite_flag= False
        dump_results = False
        see_failed = False
        save_failed = False

        if(mode=="make_conf"):
            make_confs = True

        elif(mode=='force_run'):
            make_confs = True
            run_confs  = True
            overwrite_flag= True
            dump_results = True

        elif(mode=='run'):
            make_confs = True
            run_confs  = True
            dump_results = True

        elif(mode=='save'):
            dump_results = True

        elif(mode=='failed'):
            see_failed = True
        elif(mode=='save_failed'):
            save_failed = True
        else:
            print('Specify mode: make_conf | force_run | run | save')
            exit()
    else:
        print('Specify mode: make_conf | force_run | run | save')
        exit()


    if(make_confs or run_confs):
        print(f'num post hoc confs : {len(lst_calib_confs)} ')
        print(f'num train confs : {len(lst_train_confs)} ')

        m = math.prod([len(params[k]) for k in params.keys() ])

        print(f'num global confs : {m} ')

        if(cross_train_post_hoc):
            # this will create cross product of training configs and post-hoc calibration methods.
            # will be a lot too run, so be careful in setting values above.
            params_cp = copy.deepcopy(params)

            params2 = { 'training_conf'     : lst_train_confs,
                        'calib_conf'        : lst_calib_confs}
            params_cp.update(params2)

            lst_confs          = create_confs(base_conf,params_cp)

        elif(add_train_post_hoc):

            # this will create configs with training configs + no (default) post-hoc calibration
            # and configs with default training config and all post-hoc configs created above.

            params_cp = copy.deepcopy(params)

            params_cp.update({ 'training_conf'     : lst_train_confs, 'calib_conf': [None]})
            lst_confs_1          = create_confs(base_conf,params_cp)

            params_cp = copy.deepcopy(params)

            params_cp.update( { 'training_conf': std_xent_train_confs, 'calib_conf'        : lst_calib_confs} )
            lst_confs_2          = create_confs(base_conf,params_cp)

            lst_confs = lst_confs_1 + lst_confs_2

        print(f'Total Confs to run {len(lst_confs)}')

    if(run_confs):
        run_seq = apply_conf_intel(lst_confs )
        print(len(run_seq), [len(run_seq_x) for run_seq_x in run_seq  ])

        for lst_confs_shard in run_seq:
            batched_par_run(lst_confs_shard,batch_size=run_batch_size, lst_devices=lst_devices, overwrite=overwrite_flag)

    if(dump_results):
        keys = ['calib_conf','training_conf','C_1', 'eps','max_num_train_pts','max_num_val_pts','method','query_batch_frac','seed_frac'] + extra_keys
        #keys+= list(top_lbl_hb_params.keys()) + list(scaling_params.keys()) + list(auto_lbl_opt_v2_params.keys())
        print(keys)
        save_results(root_pfx,base_conf['output_root'],keys)

    if(see_failed):
        get_failed_configs(base_conf['output_root'])

    if(save_failed):
        save_failed_configs(root_pfx, base_conf['output_root'])

