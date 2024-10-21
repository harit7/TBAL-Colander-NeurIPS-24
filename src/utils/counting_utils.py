from sklearn.metrics import accuracy_score
import logging
import sys

import torch 
# read the outputs and create a dataframe
import os 
import pickle 
import pandas as pd 
import numpy as np 

from collections import defaultdict 
import copy 

def load_pkl_file(fpath):
    with open(fpath, 'rb') as handle:
        o = pickle.load(handle)
    return o 

def get_all_outs_for_exp(root_pfx,include_nan_auto_err=True):

    lst_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(root_pfx) for f in fn]
    lst_out_files = [f  for f in lst_files if f[-3:]=='pkl'] 
    print(f'Total output pkl files read : {len(lst_out_files)}')

    lst_outs = []
    for fpath in lst_out_files:
        out = load_pkl_file(fpath)
        out_ = {} 
        params =fpath[len(root_pfx)+1:]# TODO: causing error during plotting extraction. Immediate fix: params =fpath[len(root_pfx):] breaks script runs
        params = dict([x.split('__') for x in params.split('/')[:-1] ])
        #print(fpath)
        if(params['method']=='tbal'):
            out_['sel_auto_labeled_acc'] = out['sel_counts']['auto_labeled_acc']
            out_['sel_coverage'] = out['sel_counts']['coverage_1']
            out_['all_auto_labeled_acc'] = out['sel_counts']['auto_labeled_acc']
            out_['all_coverage'] = out['sel_counts']['coverage_1']
            #print(out)
            out_['avg_ece_on_val']   = out['sel_counts']['avg_ece_on_val']
        else:
            out_['sel_auto_labeled_acc'] = out['sel_counts']['auto_labeled_acc']
            out_['sel_coverage'] = out['sel_counts']['coverage_1']
            out_['all_auto_labeled_acc'] = out['all_counts']['auto_labeled_acc']
            out_['all_coverage'] = out['all_counts']['coverage_1']

        out_.update(params)

        if(out_['sel_auto_labeled_acc']!=None):
            lst_outs.append(out_)

        elif(include_nan_auto_err):
            out_['sel_auto_labeled_acc'] = 1.0
            lst_outs.append(out_)


        #lst_outs.append(out_)
    print(f'total outs read : {len(lst_outs)}')

    return lst_outs 

def filter_outputs(lst_outs,param_f):
    filtered_outs = []
    for out in lst_outs:
        flag = True 
        for k in param_f.keys():
            flag = flag and (k in out) and (out[k]==str(param_f[k]))
        if(flag):
            filtered_outs.append(out)
    return filtered_outs

def filter_outputs_2(df,param_f):
    query = ' & '.join([ str(param)+ '==' + "'"+str(param_f[param])+"'" for param in param_f.keys()])
    return df.query(query)

def get_numbers_for_param(lst_outs,base_params,param, param_vals):
    out = defaultdict(list)

    for n in param_vals:
        
        print(n)
        params = copy.deepcopy(base_params)

        params[param] = n
        df_1 = pd.DataFrame(lst_outs)
        
        params['method'] = 'active_learning'
        #filterd_outs = filter_outputs(lst_outs,params)
        df = filter_outputs_2(df_1,params)
        #df = pd.DataFrame(filterd_outs)
        #print(df['sel_auto_labeled_acc'].mean())
        #print(df['sel_coverage'].mean())
        out['max_num_train_pts'].append(n)

        out['AL_all_err_mean'].append(1- df['all_auto_labeled_acc'].mean())
        out['AL_all_err_std'].append(df['all_auto_labeled_acc'].std())

        out['AL_all_cov_mean'].append(df['all_coverage'].mean())
        out['AL_all_cov_std'].append(df['all_coverage'].std())

        out['AL_sel_err_mean'].append(1- df['sel_auto_labeled_acc'].mean())
        out['AL_sel_err_std'].append(df['sel_auto_labeled_acc'].std())

        out['AL_sel_cov_mean'].append(df['sel_coverage'].mean())
        out['AL_sel_cov_std'].append(df['sel_coverage'].std())


        params['method'] = 'passive_learning'
        #filterd_outs = filter_outputs(lst_outs,params)
        #df = pd.DataFrame(filterd_outs)
        df = filter_outputs_2(df_1,params)

        #print(df['sel_auto_labeled_acc'].mean())
        #print(df['sel_coverage'].mean())
        out['PL_all_err_mean'].append(1- df['all_auto_labeled_acc'].mean())
        out['PL_all_err_std'].append(df['all_auto_labeled_acc'].std())

        out['PL_all_cov_mean'].append(df['all_coverage'].mean())
        out['PL_all_cov_std'].append(df['all_coverage'].std())

        out['PL_sel_err_mean'].append(1- df['sel_auto_labeled_acc'].mean())
        out['PL_sel_err_std'].append(df['sel_auto_labeled_acc'].std())
        out['PL_sel_cov_mean'].append(df['sel_coverage'].mean())
        out['PL_sel_cov_std'].append(df['sel_coverage'].std())

        params['method'] = 'active_labeling'
        #filterd_outs = filter_outputs(lst_outs,params)

        #df = pd.DataFrame(filterd_outs)
        df = filter_outputs_2(df_1,params)
        #print(df['sel_auto_labeled_acc'].mean())
        #print(df['sel_coverage'].mean())
        out['ALBL_sel_err_mean'].append(1- df['sel_auto_labeled_acc'].mean())
        out['ALBL_sel_err_std'].append(df['sel_auto_labeled_acc'].std())
        out['ALBL_sel_cov_mean'].append(df['sel_coverage'].mean())
        out['ALBL_sel_cov_std'].append(df['sel_coverage'].std())

    for k in out.keys():
        out[k] = np.array(out[k])

    return out 

def to_numpy_safely(x):
    if(type(x)== torch.Tensor):
        return x.detach().numpy()
    else:
        return x 


def get_scores_numbers(inf_out, true_labels, num_classes):
    
    true_labels = to_numpy_safely(true_labels)
    inf_out['labels'] = to_numpy_safely(inf_out['labels'])
    inf_out['confidence'] = to_numpy_safely(inf_out['confidence'])

    m = len(inf_out['confidence'])

    S = np.zeros((m,4))
    S[:,0] = inf_out['confidence']
    
    S[:,1] = inf_out['labels'] 
    S[:,2] = true_labels
    c_flags  = true_labels == inf_out['labels']
    i_flags  = true_labels != inf_out['labels']
    c_idx    = np.where(c_flags)[0]
    i_idx    = np.where(i_flags)[0]


    S2 = S[(-S[:,0]).argsort()]
    
    S_correct = S[c_idx,0]
    S_incorrect = S[i_idx,0]
    
    out = {} 
    out['correct_scores']   = S_correct
    out['incorrect_scores'] = S_incorrect
    out['correct_idx']      = c_idx 
    out['incorrect_idx']    = i_idx

    # class wise,
    class_wise_out = [] 
    for c in range(num_classes):
        cls_out = {} 
        cls_flags = inf_out['labels'] == c
        #cls_flags = inf_out['labels'] != c
        cls_c_flags = np.logical_and(cls_flags, c_flags)
        cls_i_flags = np.logical_and(cls_flags, i_flags)
        
        cls_c_idx    = np.where(cls_c_flags)[0]
        cls_i_idx    = np.where(cls_i_flags)[0]

        cls_out['correct_scores']   = S[cls_c_idx,0]
        cls_out['incorrect_scores'] = S[cls_i_idx,0]
        cls_out['correct_idx']      = cls_c_idx 
        cls_out['incorrect_idx']    = cls_i_idx

        class_wise_out.append(cls_out)

    out['class_wise_out'] = class_wise_out 
    
    return out 



from collections import defaultdict 
from datetime import datetime 
import itertools

def agg_on_seed(keys,lst_outs):
    def get_agg(sub_lst_outs):
        df_tmp = pd.DataFrame(sub_lst_outs)
        o = {}
        o['Auto-Labeling-Err-Mean'] = (1-df_tmp['sel_auto_labeled_acc']).mean()
        o['Auto-Labeling-Err-Std']  = (1-df_tmp['sel_auto_labeled_acc']).std()
        o['Coverage-Mean']          = df_tmp['sel_coverage'].mean()
        o['Coverage-Std']           = df_tmp['sel_coverage'].std()

        if('avg_ece_on_val' in df_tmp.columns):
            o['Avg-ECE-Val-Mean']       = df_tmp['avg_ece_on_val'].mean()
            o['Avg-ECE-Val-Std']        = df_tmp['avg_ece_on_val'].std()
        else:
            o['Avg-ECE-Val-Mean'] = None 
            o['Avg-ECE-Val-Std']  = None 

        
        for k in o.keys():
            o[k] = np.round(o[k]*100,4) if o[k] is not None else None 

        o['num_runs']               = len(sub_lst_outs)
        
        return o 

    D = defaultdict(list)
    D_ = {} # key: string config signature, value: config 
    for o in lst_outs:
        out_key = "##".join([f"{k}__{o[k]}" for k in keys if k in o.keys()]) # Define key
        D[out_key].append(o)
        D_[out_key] = dict([(k,o[k]) for k in keys if k in o.keys() ])

    D2 = {}
    for k in D.keys():
        D2[k] = get_agg(D[k])
        #print(k, len(D[k]))
    lst_final = []
    for k in sorted(D2.keys()):
        o = D_[k]
        o.update(D2[k])

        lst_final.append(o)
    df_agg = pd.DataFrame(lst_final)
    return df_agg


# This function finds all the failed configs in the outputs directory
def get_failed_configs(outputs_path):
    failed_configs = []
    print('######################## FAILED RUNS ########################')
    for dp, dn, fn in os.walk(outputs_path): 
        # If there are any log files, but no pkl files, then the experiment failed
        # However, this may not discern from currently running files
        if any([f.endswith('.log') for f in fn]) and not any([f.endswith('.pkl') for f in fn]):
            failed_configs.append((dp.replace(' ', r'\ ').replace(':', r'\:')))
    for num, f in enumerate(list(set(failed_configs))):
        print(f'{num+1}) {f}')
    print(f'You have {len(failed_configs)} failed configs')
    print(f'Note: Some configs might still be running, but captured as a failure.')
    print('#############################################################')
    return failed_configs

# This function saves the failed configs to a new directory: ../failed_configs
def save_failed_configs(root_pfx, outputs_path):
    f_configs = get_failed_configs(outputs_path)
    # Create a new directory at ../failed_configs to save the failed configs
    failed_configs_path = os.path.join('..', 'failed_configs', f'{root_pfx}_failed_configs')
    if not os.path.exists(failed_configs_path):
        os.makedirs(failed_configs_path)
    # Copy the failed configs path and append its file name: run_config.yaml to the new directory
    for num, f in enumerate(f_configs):
        os.system(f'cp {f}/run_config.yaml {failed_configs_path}/run_config{num}.yaml')
    print(f'Failed configs copied to {failed_configs_path}')
    print('#############################################################')


'''
This function 
1. Traverses the outputs directory specified by the root prefix and outputs path
and extracts all leaf node .pkl (pickle) files. Individual pickle files contains important
information (coverage and accuracy) for the evaluation of TBAL.
2. Aggregates information of runs across its respective random seeds for the purpose of 
depicting resultant mean and standard deviation of the metrics of interest.
3. We then consolidate these information along with other selected hyper-parameters,
and stored in a single excel file for easier analysis.
'''
def save_results(root_pfx,outputs_path,keys,include_nan_auto_err=True):

    lst_outs = get_all_outs_for_exp(outputs_path,include_nan_auto_err=include_nan_auto_err)
    
    print('total outs: ', len(lst_outs))

    df     = pd.DataFrame(lst_outs)

    df_agg = agg_on_seed(keys,lst_outs)
    
    print('Size of total agg df: ',len(df_agg))

    rename_cols = { "max_num_train_pts":"N_t", 
                    "max_num_val_pts": "N_v" ,
                    "num_hyp_val_samples": "N_hyp_v",
                    "training_conf_g.learning_rate": "lr_g", 
                    "training_conf_t.learning_rate" : "lr_t"
                    }
    df_agg = df_agg.rename(columns=rename_cols) 
    df     = df.rename(columns = rename_cols)

    # Writing to xlsx file in different sheets
    dt_string = datetime.now().strftime("%m-%d-%Y__%H-%M-%S")

    res_file_name = f"{root_pfx}__{dt_string}.xlsx"
    res_file_path = os.path.join(outputs_path,res_file_name)

    writer = pd.ExcelWriter(res_file_path) 
    all_columns = list(df_agg.columns)
    all_columns.sort()

    columns = ["calib_conf","training_conf",'C_1',"N_t","N_v", "N_hyp_v"]+\
            ["Auto-Labeling-Err-Mean","Coverage-Mean", "Avg-ECE-Val-Mean"] +\
            [ "Auto-Labeling-Err-Std" ,"Coverage-Std" , "Avg-ECE-Val-Std"]

    columns = columns + [c for c in all_columns if c not in columns]

    print('****')
    print(columns)

    sheets_on_columns = ['C_1','N_t','N_v']
    #print(df_agg)

    #df_agg.sort_values(by=['Coverage-Mean', 'Coverage-Std'], ascending=[False, True], inplace=True)
    #print(df_agg.index.duplicated())
    print(df_agg[df_agg.index.duplicated()])

    df_agg.to_excel(writer, sheet_name='All_Agg_Results', index=True, na_rep='NaN', columns=columns)

    sheets_on_columns = ['C_1','N_t','N_v']
    lsts = [list(df_agg[c].unique()) for c in sheets_on_columns]

    sheet_names = ["All_Agg_Results"]
    

    for element in itertools.product(*lsts):
        sheet_name = "__".join([f"{sheets_on_columns[i]}_{element[i]}" for i in range(len(element)) ])
        q = ' & '.join([f"{sheets_on_columns[i]} == '{element[i]}'" for i in range(len(element)) ])
        df_agg_filtered = df_agg.query(q)
        df_agg_filtered.to_excel(writer, sheet_name=sheet_name, index=False, na_rep='NaN', columns=columns)
        sheet_names.append(sheet_name)

    df.to_excel(writer,sheet_name='All_Results',index=False,na_rep='NaN')

    # update column sizes
    for i, column in enumerate(columns):
        column_length = max(df_agg[column].astype(str).map(len).max(), len(column))
        column_length = max(int(column_length*1.2),8)
        #col_idx = df_agg.columns.get_loc(column)
        col_idx = i 
        #print(column, column_length, col_idx)
        for sheet_name in sheet_names:
            writer.sheets[sheet_name].set_column(col_idx, col_idx, column_length)
    writer.close()

    print('Saved results at path : ', res_file_path)
