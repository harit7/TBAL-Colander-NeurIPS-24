import sys
from multiprocessing import Process
from os.path import join

sys.path.append('../')
from omegaconf import OmegaConf
from src.utils.run_lib import *
from src.utils.conf_utils import  * 

import pandas as pd 

from datetime import datetime 

calib_val_frac = 0.0025

def augment_conf(conf):
    calib_conf = conf['calib_conf']

    conf['run_dir']        = join(conf['output_root'], calib_conf['name'])
    conf['log_file_path']  = join(conf['run_dir'], conf['method'] + '.log')
    conf['out_file_path']  = join(conf['run_dir'], conf['method'] + '.pkl')
    conf['conf_save_path'] = join(conf['run_dir'], conf['method'] + '.yaml')

    if(calib_conf['type']=='post_hoc'):
        ckpt_file_name = get_model_ckpt_file_name(conf)

        conf.training_conf['save_ckpt'] = True 
        conf.training_conf['train_from_scratch'] = False  
        conf.training_conf['ckpt_save_path'] = join(root_dir, 'ckpt', ckpt_file_name)
        conf.training_conf['ckpt_load_path'] = join(root_dir, 'ckpt', ckpt_file_name)

    if(calib_conf['type']=='train_time'):
        for k in  calib_conf.training_conf.keys():
            conf.training_conf[k] = calib_conf.training_conf[k]

        ckpt_file_name = get_model_ckpt_file_name(conf)

        conf.training_conf['ckpt_save_path'] = f'{root_dir}/ckpt/{ckpt_file_name}'


    return conf 

def test_xent(conf,conf_dir,all_outs,stdout=False):
    conf['calib_conf']     =  OmegaConf.load(f'{conf_dir}/train-time/xent_calib_conf.yaml')
    calib_conf = conf['calib_conf']

    augment_conf(conf)
    out= run_conf(conf,stdout=stdout)

    o   = out['sel_counts'] 
    o['method'] = calib_conf['name']
    all_outs.append(o)

    return out 

def test_squentropy(conf, conf_dir, all_outs,stdout=False):
    conf['calib_conf']     =  OmegaConf.load(f'{conf_dir}/train-time/squentropy_calib_conf.yaml')
    calib_conf = conf['calib_conf']

    augment_conf(conf)
    out = run_conf(conf,stdout=stdout)
    o   = out['sel_counts'] 
    o['method'] = calib_conf['name']
    all_outs.append(o)
    
    return out 

def test_label_smoothing(conf,conf_dir,all_outs, stdout=False):
    conf['calib_conf']     =  OmegaConf.load(f'{conf_dir}/train-time/label_smoothing_calib_conf.yaml')
    
    calib_conf = conf['calib_conf']
    calib_conf.training_conf['label_smoothing'] = 0.15
    
    augment_conf(conf)
    out = run_conf(conf,stdout=stdout)
    o   = out['sel_counts'] 
    o['method'] = calib_conf['name']
    all_outs.append(o)
    return out 

def test_focal_loss(conf,conf_dir,all_outs,stdout=False):
    conf['calib_conf']     =  OmegaConf.load(f'{conf_dir}/train-time/focal_calib_conf.yaml')
    params                 =  { "gamma": [2.0] }
    calib_conf = conf['calib_conf']

    augment_conf(conf)
    out = run_conf(conf,stdout=stdout)
    o   = out['sel_counts'] 
    o['method'] = calib_conf['name']
    all_outs.append(o)

    return out 
        
def test_crl(conf,conf_dir, all_outs,stdout=False):
    conf['calib_conf']     =  OmegaConf.load(f'{conf_dir}/train-time/crl_calib_conf.yaml')
    
    calib_conf = conf['calib_conf']
    calib_train_conf = conf.calib_conf.training_conf 
    calib_train_conf['rank_target'] = 'margin' ## options : softmax, margin, entropy
    calib_train_conf['rank_weight'] = 0.01

    augment_conf(conf)
    out = run_conf(conf,stdout=stdout)
    o   = out['sel_counts'] 
    o['method'] = calib_conf['name']
    all_outs.append(o)
    return out 

def test_mixup(conf,conf_dir,all_outs, stdout=False):
    conf['calib_conf']     =  OmegaConf.load(f'{conf_dir}/train-time/mixup_calib_conf.yaml')

    calib_train_conf = conf.calib_conf.training_conf 
    calib_train_conf['mixup_alpha'] = 0.9
    calib_conf = conf['calib_conf']
    augment_conf(conf)
    out = run_conf(conf,stdout=stdout)
    o   = out['sel_counts'] 
    o['method'] = calib_conf['name']
    all_outs.append(o)
    return out 

def test_mmce(conf,conf_dir, all_outs, stdout=False):
    conf['calib_conf']     =  OmegaConf.load(f'{conf_dir}/train-time/mmce_calib_conf.yaml')
    
    calib_conf = conf['calib_conf']
    calib_train_conf = calib_conf.training_conf 
    calib_train_conf['mmce_coeff'] = 0.055

    augment_conf(conf)
    out = run_conf(conf,stdout=stdout)
    o   = out['sel_counts'] 
    o['method'] = calib_conf['name']
    all_outs.append(o)
    return out 

def test_fmfp(conf,conf_dir,all_outs, stdout=False):
    conf['calib_conf']     =  OmegaConf.load(f'{conf_dir}/train-time/fmfp_calib_conf.yaml')
    #calib_train_conf = conf.calib_conf.training_conf
    calib_conf = conf['calib_conf'] 
    augment_conf(conf)
    out = run_conf(conf,stdout=stdout)
    o   = out['sel_counts'] 
    o['method'] = calib_conf['name']
    all_outs.append(o)
    return out 


def test_top_lbl_hb(conf,conf_dir, all_outs, stdout=False):
    conf['calib_conf']      = OmegaConf.load(f'{conf_dir}/post-hoc/top_label_hist_bin_base_conf.yaml')
    calib_conf = conf['calib_conf']
    calib_conf['points_per_bin'] = 25
    calib_conf['calib_val_frac'] = calib_val_frac

    augment_conf(conf)
    out = run_conf(conf,stdout=stdout)
    o   = out['sel_counts'] 
    o['method'] = calib_conf['name']
    all_outs.append(o)
    return out 


def test_temp_scaling(conf,conf_dir, all_outs, stdout=False):
    conf['calib_conf']      = OmegaConf.load(f'{conf_dir}/post-hoc/scaling_conf.yaml')
    calib_conf = conf['calib_conf']
    calib_conf['calib_val_frac'] = calib_val_frac

    t_conf = conf['calib_conf']['training_conf']
    t_conf['optimizer']  = 'sgd'
    t_conf['learning_rate'] = 0.001
    t_conf['weight_decay'] = 0.01
    t_conf['max_epochs'] = 500

    augment_conf(conf)
    out = run_conf(conf,stdout=stdout)
    o   = out['sel_counts'] 
    o['method'] = calib_conf['name']
    all_outs.append(o)
    return out 

def test_scaling_binning(conf,conf_dir,all_outs, stdout=False):
    conf['calib_conf']      = OmegaConf.load(f'{conf_dir}/post-hoc/scaling_binning_base_conf.yaml')
    calib_conf = conf['calib_conf']
    calib_conf['calib_val_frac'] = calib_val_frac
    
    augment_conf(conf)

    out = run_conf(conf,stdout=stdout)

    o   = out['sel_counts'] 
    o['method'] = calib_conf['name']
    all_outs.append(o)
    return out 

def test_dirichlet(conf,conf_dir,all_outs,stdout=False):
    conf['calib_conf']      = OmegaConf.load(f'{conf_dir}/post-hoc/dirichlet_base_conf.yaml')
    calib_conf = conf['calib_conf']
    calib_conf['calib_val_frac'] = calib_val_frac
    augment_conf(conf)

    out = run_conf(conf,stdout=stdout)

    o   = out['sel_counts'] 
    o['method'] = calib_conf['name']
    all_outs.append(o)
    return out 



def test_auto_label_opt_v0(conf,conf_dir,all_outs,stdout=False):
    
    conf['calib_conf'] = OmegaConf.load(f'{conf_dir}/post-hoc/auto_lbl_opt_v0_conf.yaml')
    
    calib_conf = conf['calib_conf']

    calib_conf['l1']=1.0 
    calib_conf['l2']=10.0 # 3 or 5
    calib_conf['l3']=0.0  #1.0
    calib_conf['l4']=0.0
    calib_conf['calib_val_frac'] = calib_val_frac 
    
    calib_conf['features_key']  = 'concat'
    #calib_conf['features_key']  = 'logits'
    #calib_conf['features_key']  = 'pre_logits'

    #calib_conf['class_wise'] = 'joint'
    calib_conf['class_wise'] = 'independent'
    #calib_conf['class_wise'] ="joint_g_independent_t"
    model = "two_layer"
    #model = "linear" 

    if(model=="two_layer"):
        model_conf = OmegaConf.load('{}/model_confs/two_layer_net_base_conf.yaml'.format(conf_dir))
    
        model_conf.layers[0]['dim_factor']=2
        model_conf.layers[1]['act_fun']='tanh'

    else:
        model_conf = {} 

    calib_conf['num_classes'] = 20 

    model_conf['num_classes'] = conf.data_conf.num_classes

    calib_conf['model_conf'] = model_conf 


    calib_conf['regularize'] = False  
    calib_conf['auto_lbl_conf'] = conf.auto_lbl_conf

    
    #calib_conf['features_key']  = 'logits'

    calib_conf['alpha_1'] =0.1 #/1.5
    calib_conf['training_conf_g']['batch_size'] = 64  # 64

    lr = 1e-1

    calib_conf['training_conf_g']['optimizer'] = 'sgd'
    calib_conf['training_conf_g']['learning_rate'] = lr
    calib_conf['training_conf_g']['weight_decay'] = 1.0

    calib_conf['training_conf_t']['optimizer'] = 'sgd'
    calib_conf['training_conf_t']['learning_rate'] =lr
    calib_conf['training_conf_t']['weight_decay'] = 1.0


    calib_conf['training_conf_g']['max_epochs'] = 500

    augment_conf(conf)
    
    out = run_conf(conf,stdout=stdout)

    o   = out['sel_counts'] 
    o['method'] = calib_conf['name']
    all_outs.append(o)

    return out 




def test_auto_label_opt_v2(conf,conf_dir,all_outs,stdout=False):
    
    conf['calib_conf'] = OmegaConf.load(f'{conf_dir}/post-hoc/auto_lbl_opt_v2_calib.yaml')
    
    calib_conf = conf['calib_conf']

    calib_conf['l1']=1.0
    calib_conf['l2']=1.0
    calib_conf['l3']=15.0
    calib_conf['l4']=1.0
    calib_conf['calib_val_frac'] = calib_val_frac

    calib_conf['regularize'] = True  
    calib_conf['auto_lbl_conf'] = conf.auto_lbl_conf


    calib_conf['features_key']  = 'pre_logits'

    calib_conf['alpha_1'] = 0.5 #/1.5
    calib_conf['training_conf_g']['batch_size'] = 500

    calib_conf['training_conf_g']['optimizer'] = 'adam'
    calib_conf['training_conf_g']['learning_rate'] = 0.0001
    calib_conf['training_conf_g']['weight_decay'] = 0.0001

    calib_conf['training_conf_g']['max_epochs'] = 2000

   

    model_conf = {} 

    model_conf['num_classes'] = conf.data_conf.num_classes
    model_conf['name'] = 'linear'
    
    model_conf['layers'] = []
    calib_conf['model_conf'] = model_conf 

    augment_conf(conf)
    
    out = run_conf(conf,stdout=stdout)

    o   = out['sel_counts'] 
    o['method'] = calib_conf['name']
    all_outs.append(o)

    return out 


if __name__ == "__main__":

    root_dir = '../'
    conf_dir = f'{root_dir}configs/calib-exp/'
    
    print(sys.argv)

    if(len(sys.argv)>1):
        model_ds_key = sys.argv[1]
        method       = sys.argv[2]
    else:
        # model_ds_key = 'svhn_simplenet'
        #model_ds_key = 'mnist_lenet'
        #model_ds_key = 'cifar10_resnet18'
        #model_ds_key = 'tiny_imagenet_CLIP'
        model_ds_key = 'twenty_newsgroups'

        method = 'passive_learning'
        #method = 'active_labeling'
    
    print(model_ds_key, method)

    root_pfx = f'test_runs/{model_ds_key}_calib/'

    base_conf                = OmegaConf.load(f'{conf_dir}/{model_ds_key}_base_conf.yaml')
    base_conf['output_root'] = join(root_dir, 'outputs', root_pfx)

    #base_conf['eval'] = 'hyp'
    base_conf['eval'] = 'full'
    base_conf['random_seed']= 1

    base_conf['method'] = method #'passive_learning' # single round
    #base_conf['method'] = 'active_labeling'  # multi round

    base_conf['data_conf']['compute_emb'] = False 

    base_conf['val_pts_query_conf']['max_num_val_pts'] = 400

    base_conf['train_pts_query_conf']['max_num_train_pts'] = 400
    
    run_train_time = True 
    #run_train_time = False  

    run_post_hoc   = True  

    all_outs = []
    if(run_train_time):
        test_xent(base_conf,conf_dir,all_outs, stdout=True) 
        '''
        test_squentropy(base_conf,conf_dir,all_outs,stdout=True)
        
        test_label_smoothing(base_conf,conf_dir,all_outs,stdout=True)

        test_focal_loss(base_conf,conf_dir,all_outs,stdout=True)

        test_crl(base_conf,conf_dir,all_outs,stdout=True)

        test_mixup(base_conf, conf_dir, all_outs,stdout=True)

        test_mmce(base_conf, conf_dir, all_outs,stdout=True)
        
        test_fmfp(base_conf, conf_dir, all_outs,stdout=True)
        '''

    if(run_post_hoc):
        #test_auto_label_opt_v2(base_conf,conf_dir,all_outs,stdout=True)

        test_auto_label_opt_v0(base_conf,conf_dir,all_outs,stdout=True)
        
        #test_top_lbl_hb(base_conf,conf_dir,all_outs,stdout=True)

        #test_temp_scaling(base_conf,conf_dir,all_outs,stdout=True)

        #test_scaling_binning(base_conf,conf_dir,all_outs,stdout=True)

        #test_dirichlet(base_conf,conf_dir,all_outs,stdout=True)
        

    df_all = pd.DataFrame(all_outs)
    print(df_all)

    dt_string = datetime.now().strftime("%m-%d-%Y__%H-%M-%S")

    res_file_path = join(root_dir, 'outputs', f'test_runs_{method}_{model_ds_key}__{dt_string}.csv')
    

    #df_all.to_csv(res_file_path, index=False, columns=['method', 'auto_labeled_acc', 'coverage_1'])


    


    

