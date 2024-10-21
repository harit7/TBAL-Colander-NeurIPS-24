import os
import sys 
import copy
import time
import random 
from multiprocessing import Process
from omegaconf import OmegaConf
# import wandb

sys.path.append('../')
sys.path.append('../../')

from src.core.conf_defaults import *
from src.core.passive_learning import *
from src.core.tbal import * 
from src.core.active_learning import * 
from src.core.auto_labeling import *
from src.core.self_training import * 

from src.data_layer.dataset_utils import * 
from src.data_layer.data_manager import * 

from src.utils.counting_utils import *  
from src.utils.common_utils import * 

from src.utils.vis_utils import *
from src.utils.logging_utils import * 

from .common_utils import set_seed


def run_active_learnling_auto_labeling(conf,logger):

    set_seed(conf['random_seed'])
    # get data
    dm = DataManager(conf,logger, lib=conf['model_conf']['lib'])
    logger.info('Loaded dataset {}'.format(conf['data_conf']['name']))
    logger.info(f' std_train_size : {len(dm.ds_std_train)} and  std_val_size: {len(dm.ds_std_val)}')
    

    act_learn = ActiveLearning(conf,dm,logger)

    out = act_learn.run_al_loop()

    auto_lbl_conf = conf['auto_lbl_conf']

    auto_lbl_conf['method_name']= 'all' 
    
    meta_df_cp = copy.deepcopy(dm.meta_df) 

    auto_labeler = AutoLabeling(conf,dm,act_learn.cur_clf,logger)
    out = auto_labeler.run()
    counts_all = dm.get_auto_labeling_counts()

    auto_lbl_conf['method_name']= 'selective' 
    
    dm.meta_df = meta_df_cp 

    auto_labeler = AutoLabeling(conf,dm,act_learn.cur_clf,logger)
    out = auto_labeler.run()
    counts_sel = dm.get_auto_labeling_counts()

    return {'all_counts':counts_all,'sel_counts':counts_sel}

def run_passive_label_all(conf, logger):
    set_seed(conf['random_seed'])
    # get data
    dm = DataManager(conf,logger, lib=conf['model_conf']['lib'])

    logger.info('Loaded dataset {}'.format(conf['data_conf']['name']))
    logger.info(f' std_train_size : {len(dm.ds_std_train)} and  std_val_size: {len(dm.ds_std_val)}')

    pas_learn = PassiveLearning(conf,dm,logger)

    out = pas_learn.run()

    auto_lbl_conf = conf['auto_lbl_conf']

    auto_lbl_conf['method_name']= 'all' 
    
    meta_df_cp = copy.deepcopy(dm.meta_df) 

    auto_labeler = AutoLabeling(conf,dm,pas_learn.cur_clf,logger)
    out = auto_labeler.run()
    counts_all = dm.get_auto_labeling_counts()
    logger.info(f" All Auto-labeling counts: {counts_all}")


def run_al_self_training_auto_labeling_all(conf,logger):
    set_seed(conf['random_seed'])
    # get data
    dm = DataManager(conf,logger, lib=conf['model_conf']['lib'])

    logger.info('Loaded dataset {}'.format(conf['data_conf']['name']))
    logger.info(f' std_train_size : {len(dm.ds_std_train)} and  std_val_size: {len(dm.ds_std_val)}')
    
    print(len(dm.ds_std_train), len(dm.ds_std_val), len(dm.ds_std_test), len(dm.ds_hyp_val))

    st = SelfTraining(conf,dm,logger)

    out = st.run_al_loop()

    auto_lbl_conf = conf['auto_lbl_conf']

    auto_lbl_conf['method_name']= 'all' 
    
    #meta_df_cp = copy.deepcopy(dm.meta_df) 

    auto_labeler = AutoLabeling(conf,dm, st.cur_clf,logger, None)
    out = auto_labeler.run()
    counts_all = dm.get_auto_labeling_counts()
    logger.info(f" All Auto-labeling counts: {counts_all}")

    counts_all['avg_ece_on_val'] =  out['ECE_on_val'] if 'ECE_on_val' in out else None
    counts_all['avg_ece_no_calib_on_val'] = out['ECE_on_val_no_calib'] if 'ECE_on_val_no_calib' in out else None
    
    
    return {'all_counts':counts_all,'sel_counts':counts_all}



def run_al_self_training_auto_labeling(conf,logger):
    set_seed(conf['random_seed'])
    # get data
    dm = DataManager(conf,logger, lib=conf['model_conf']['lib'])

    logger.info('Loaded dataset {}'.format(conf['data_conf']['name']))
    logger.info(f' std_train_size : {len(dm.ds_std_train)} and  std_val_size: {len(dm.ds_std_val)}')
    
    print(len(dm.ds_std_train), len(dm.ds_std_val), len(dm.ds_std_test), len(dm.ds_hyp_val))

    st = SelfTraining(conf,dm,logger)

    out = st.run_al_loop()

    auto_lbl_conf = conf['auto_lbl_conf']

    auto_lbl_conf['method_name']= 'selective' 


    auto_labeler = AutoLabeling(conf,dm,st.cur_clf,logger, None)
    out = auto_labeler.run()
    counts_sel = dm.get_auto_labeling_counts()

    
    logger.info(f" Selective Auto-labeling counts: {counts_sel}")

    counts_sel['avg_ece_on_val'] =  out['ECE_on_val'] if 'ECE_on_val' in out else None
    counts_sel['avg_ece_no_calib_on_val'] = out['ECE_on_val_no_calib'] if 'ECE_on_val_no_calib' in out else None
    
    
    return {'all_counts':counts_sel,'sel_counts':counts_sel}


def run_passive_learning_auto_labeling(conf,logger):

    set_seed(conf['random_seed'])
    # get data
    dm = DataManager(conf,logger, lib=conf['model_conf']['lib'])

    logger.info('Loaded dataset {}'.format(conf['data_conf']['name']))
    logger.info(f' std_train_size : {len(dm.ds_std_train)} and  std_val_size: {len(dm.ds_std_val)}')
    
    print(len(dm.ds_std_train), len(dm.ds_std_val), len(dm.ds_std_test), len(dm.ds_hyp_val))

    pas_learn = PassiveLearning(conf,dm,logger)

    out = pas_learn.run()

    auto_lbl_conf = conf['auto_lbl_conf']

    auto_lbl_conf['method_name']= 'selective' 

    #<<<<<<<<<<<<<<<<<<<<<<<<< BEGIN CALIBRATION BLOCK <<<<<<<<<<<<<<<<<<<<<<<<<

    if( conf['calib_conf'] and conf['calib_conf']['type']=='post_hoc'):
        calib_conf    = conf['calib_conf'] 
        logger.info('========================= Training Post-hoc Calibrator   =========================')
        
        logger.info(f"Calib Conf : {conf['calib_conf']}")

        cur_calibrator  = get_calibrator(pas_learn.cur_clf,calib_conf,logger)

        # randomly split the current available validation points into two parts.
        # one part will be used for training the calibrator and other part for finding 
        # the auto-labeling thresholds.
        dm.select_calib_val_points(calib_frac=calib_conf['calib_val_frac'])
        
        cur_val_ds_c , cur_val_idcs_c    = dm.get_cur_calib_val_ds()
        cur_val_ds_nc , cur_val_idcs_nc    = dm.get_cur_non_calib_val_ds()

        #print(np.histogram(cur_val_ds_c.Y.numpy()))

        values, counts = np.unique(cur_val_ds_c.Y.numpy(), return_counts=True)
        #print(values)
        #print(counts)

        logger.info(f"Number of validation points for training calibrator : {len(cur_val_idcs_c)}")
        cur_calibrator.fit(cur_val_ds_c,ds_val_nc=cur_val_ds_nc)
    else:
        logger.info('=========================    No Post-hoc Calibration     =========================')
        cur_calibrator = None 
    
    #>>>>>>>>>>>>>>>>>>>>>>>>>>> END CALIBRATION BLOCK  >>>>>>>>>>>>>>>>>>>>>>>>>>>
    
    auto_labeler = AutoLabeling(conf,dm,pas_learn.cur_clf,logger,cur_calibrator)
    out = auto_labeler.run()
    counts_sel = dm.get_auto_labeling_counts()

    
    logger.info(f" Selective Auto-labeling counts: {counts_sel}")

    counts_sel['avg_ece_on_val'] =  out['ECE_on_val'] if 'ECE_on_val' in out else None
    counts_sel['avg_ece_no_calib_on_val'] = out['ECE_on_val_no_calib'] if 'ECE_on_val_no_calib' in out else None
    
    
    return {'all_counts':counts_sel,'sel_counts':counts_sel}

    #return #{'all_out':out_all,'all_counts':counts_all,'sel_out':out_sel,'sel_counts':counts_sel}

def run_tbal_conf(conf,logger,return_per_epoch_out=False):

    set_seed(conf['random_seed'])

    dm = DataManager(conf,logger,lib=conf['model_conf']['lib'])

    logger.info('Loaded dataset {}'.format(conf['data_conf']['name']))
    logger.info(f' std_train_size : {len(dm.ds_std_train)} and  std_val_size: {len(dm.ds_std_val)}')

    act_lbl = ThresholdBasedAutoLabeling(conf,dm,logger)

    act_lbl.init()

    lst_epoch_out = act_lbl.run_al_loop()
    
    logger.info('AL Loop Done')
    #test_err = al.get_test_error(al.cur_clf,test_set,conf['inference_conf'])
    out =  dm.get_auto_labeling_counts()
    
    lst_epoch_out_2 = [ epoch_out for epoch_out in lst_epoch_out if 'ECE_on_val' in epoch_out]
    
    #print(len(lst_epoch_out_2), len(lst_epoch_out))

    out['avg_ece_on_val'] = np.nanmean([epoch_out['ECE_on_val'] for epoch_out in lst_epoch_out_2 ])

    lst_epoch_out_2 = [ epoch_out for epoch_out in lst_epoch_out if 'ECE_on_val_no_calib' in epoch_out]

    out['avg_ece_no_calib_on_val'] = np.nanmean([epoch_out['ECE_on_val_no_calib'] for epoch_out in lst_epoch_out_2])



    #,"epoch_outs":lst_epoch_out
    if return_per_epoch_out:
        return {"sel_counts":out ,"lst_epoch_out":lst_epoch_out, "all_counts":out}
    else:
        return {"sel_counts":out , "all_counts":out}


def run_conf(conf,overwrite=True,stdout=False):

    if('calib_conf' in conf and conf['calib_conf']):
        conf['calib_conf']['device'] = conf['device']

    if(not overwrite):
        if(os.path.exists(conf['out_file_path'])):
            print(f"path exists {conf['out_file_path']}")
            return 
    try:
        os.makedirs(conf['run_dir'])
    except OSError:
        pass
    
    ckpt_save_path = conf.training_conf['ckpt_save_path'] 

    if(ckpt_save_path) :
        #check if ckpt_save_path directories exist, if not create.
        ckpt_dir_path  = os.path.sep.join( ckpt_save_path.split(os.path.sep)[:-1])

        try:
            os.makedirs(ckpt_dir_path)
        except OSError:
            pass


    set_defaults(conf)

    conf['inference_conf']['device'] = conf['device']
    
    if('conf_file_path' in conf):
        with open(conf['conf_file_path'],'w') as f:
             OmegaConf.save(config=conf, f=f)
    
    logger = get_logger(conf['log_file_path'],stdout_redirect=stdout,level=logging.DEBUG)
    
    print(f"Running a conf with log file at : {conf['log_file_path']}")

    if(conf['method']=='tbal'):
        out = run_tbal_conf(conf,logger)
    elif(conf['method']=='active_learning'):
        out = run_active_learnling_auto_labeling(conf,logger)
    elif(conf['method']=='passive_learning'):
        out = run_passive_learning_auto_labeling(conf,logger)
    elif(conf['method']=='al_st'):
        out = run_al_self_training_auto_labeling(conf,logger)
    elif(conf['method']=='al_st_all'):
        out = run_al_self_training_auto_labeling_all(conf,logger)
    
    # Future integration: Add Weight and Biases (wandb) tracker conf (config) and out (autolabeling metrics)
    # wandb.init(
    #     project = conf['root_pfx'],
    #     config = dict(conf)
    # )
    # wandb.log(out)
    # wandb.finish()
    
    with open(conf['out_file_path'], 'wb') as out_file:
        pickle.dump(out, out_file, protocol=pickle.HIGHEST_PROTOCOL) 
    
    close_logger(logger)

    return out 

def run_conf_2(conf):
    logger = get_logger(conf['log_file_path'],stdout_redirect=False,level=logging.DEBUG)
    logger.info('Dry Run..')
    close_logger(logger)

def par_run(lst_confs,overwrite=True):
    lstP = []
    print(len(lst_confs))
    for conf in lst_confs:
        #print(conf)
        #conf = copy.deepcopy(conf) # ensure no shit happens
        p = Process(target = run_conf, args=(conf,overwrite))
        p.start()
        
        lstP.append(p)
    for p in lstP:
        p.join()

def assign_devices_to_confs(lst_confs,lst_devices = ['cpu']): 
    #round robin
    i = 0
    n = len(lst_confs)
    while(i<n):
        for dev in lst_devices:
            if(i<n):
                lst_confs[i]['device'] = dev
            i+=1
    
def exclude_existing_confs(lst_confs):
    lst_out_confs = []
    for conf in lst_confs:
        path = conf["out_file_path"]
        if os.path.exists(path):
            print(f"path exists {conf['out_file_path']}")
        else:
            lst_out_confs.append(conf)
    return lst_out_confs

def apply_conf_intel(lst_confs,method,_eval="full"):
    
    run_seq = [lst_confs]
    
    if (method == "passive_learning") or ( method == "tbal" and _eval=="hyp"):

        p_confs = [conf for conf in lst_confs if conf['method'] == method]
        o_confs   = [conf for conf in lst_confs if conf['method'] != method] 

        p_no_post_hoc_confs = [conf for conf in p_confs if conf['calib_conf'] is None]
        p_post_hoc_confs    = [conf for conf in p_confs if conf['calib_conf'] is not None]

        for p_conf in p_post_hoc_confs:
            # these are "parasite" configs, that will use the check points created by the no-post-hoc versions run before them.
            # they should not write any checkpoints.
            # only doing this for single round auto-label configs.

            p_conf.training_conf['ckpt_save_path'] = None #p_conf['no_calib_ckpt_save_path']
            p_conf.training_conf['ckpt_load_path'] = p_conf.training_conf['no_calib_ckpt_load_path']
            p_conf.training_conf['save_ckpt']      = False 
            p_conf.training_conf['train_from_scratch'] = False  

        if(len(p_no_post_hoc_confs)>0):
            run_seq = [ p_no_post_hoc_confs , p_post_hoc_confs + o_confs ]
        else:
            run_seq = [p_post_hoc_confs + o_confs ]

    return run_seq 



def batched_par_run(lst_confs,batch_size=2, lst_devices=['cpu'],overwrite=True):
    
    if(not overwrite):
        lst_confs = exclude_existing_confs(lst_confs)
        n = len(lst_confs)
        print(f'NUM confs to run : {n}')

    assign_devices_to_confs(lst_confs,lst_devices)

    i=0
    n = len(lst_confs)
    total_time = 0
    big_bang_time = time.time()
    while(i<n):
        start = time.time()
        print(f'running confs from {i} to {i+batch_size} ')
        #for conf in lst_confs[i:i+batch_size]:
        #    print(conf['device'])
        par_run(lst_confs[i:i+batch_size],overwrite)
        
        i+=batch_size 
        end = time.time()
        u = round((end-start)/60,2) # in minutes
        print( f"Time taken to run these confs : {u} minutes, ")

        total_time = round((end-big_bang_time)/60, 2) # in minutes
        avg_time   = round((total_time/i), 2)  # already in minutes
        r = round( max(0,(n - i))*avg_time, 2)

        print(f"Total confs run so far : {i}")
        print(f"Total time taken so far : {total_time//60} hours and {total_time%60 : .2f} minutes")
        print( f"Avg. Time taken to run confs so far : {avg_time//60} hours and {avg_time%60 :.2f} minutes, ")
        print(f"Remaining confs: {n-i} and estimated time to be taken : { r//60} hours and {r%60 :.2f} minutes")

def seq_run(lst_confs):
    for conf in lst_confs:
        run_conf(conf)
