import pandas as pd 
import numpy as np
import random
from sklearn.metrics import accuracy_score

from .datasets import dataset_factory
from .dataset_utils import * 

class DataManager:
    def __init__(self,conf,logger,ds=None,lib=None):
        self.conf = conf 
        self.logger = logger
        # this is the std validation dataset 
        # a subset is selected from this for evaluation.
        #set_seed(conf.random_seed)
        
        data_conf = conf.data_conf
        if(ds is None):
            # get data
            ds = dataset_factory.load_dataset(conf)
            ds.build_dataset()
        
        self.ds = ds 
        self.ds_std_test = ds.get_test_datasets()
        
        if(lib is None and 'lib' in conf):
            lib = conf['lib']
        
        if(lib=='pytorch' 
           and not isinstance(self.ds.X, list) 
           and not isinstance(self.ds_std_test.X, list)):
            self.ds = tensorize(self.ds)
            self.ds_std_test = tensorize(self.ds_std_test)

        self.meta_df = self.init_meta_df(ds)
        self.lbl_idx_map_std_train = None
        
        idcs_hyp_val = [] 

        if('val_fraction' in data_conf):
            # split the standard data pool into train and val sets
            out = randomly_split_ds_idcs(ds,fraction=data_conf.val_fraction)
            self.idcs_std_train = out['idcs_std_train']
            self.idcs_std_val   = out['idcs_std_val']
            self.ds_std_train   = out['ds_std_train']
            self.ds_std_val     = out['ds_std_val']

            if not isinstance(self.ds_std_train.X, list):
                logger.debug(self.ds_std_train.X.shape)
            
        else:
            self.idcs_std_train = ds.idcs_std_train 

            self.idcs_std_val   = ds.idcs_std_val
            self.ds_std_train   = ds.ds_std_train 
            self.ds_std_val     = ds.ds_std_val 

        if('num_hyp_val_samples' in data_conf and data_conf['num_hyp_val_samples']>0 ):
            
            nhv = data_conf.num_hyp_val_samples 
            ps = np.random.get_state()

            np.random.seed(42) # use same fixed seed for this.
                
            assert len(self.idcs_std_val) >= 2*nhv

            idcs_hyp_val     = np.random.choice(self.idcs_std_val, nhv,replace=False)
            idcs_hyp_val_set = set(list(idcs_hyp_val))

            idcs_remain_val  = np.array( list(set(self.idcs_std_val).difference(idcs_hyp_val_set) ))
    
            self.ds_std_val , self.ds_hyp_val   = ds.get_subset(idcs_remain_val), ds.get_subset(idcs_hyp_val)   

            np.random.set_state(ps) 
            
            self.idcs_std_val = idcs_remain_val

        self.num_classes = ds.num_classes

        self.meta_df = self.init_meta_df(ds)
        for idx in self.idcs_std_train:
            self.meta_df.at[idx,'is_std_train']= True
        
        for idx in self.idcs_std_val:
            self.meta_df.at[idx,'is_std_val'] = True

        for idx in idcs_hyp_val:
            
            self.meta_df.at[idx,'is_hyp_val'] = True


        logger.info('Loaded dataset {}'.format(conf['data_conf']['name']))
        logger.info(f'Std train size: {len(self.ds_std_train)} and Std. Val. Size:{len(self.ds_std_val)}') 
    
    def init_meta_df(self,ds):
        n = len(ds)
        meta_df = pd.DataFrame({'idx':range(n)})
        meta_df['is_labeled']    = False
        meta_df['confidence']    = 1.0
        meta_df['true_label']    = ds.Y
        
        meta_df['auto_label']  = -1
        meta_df['is_auto_labeled']  = False 
        meta_df['auto_labeled_in_round'] = -1 
        
        meta_df['pseudo_label']  = -1
        meta_df['is_pseudo_labeled']  = False

        #meta_df['human_labeled'] = False

        meta_df['queried_in_round']=-1
        meta_df['is_queried']    = False 
        
        # is_queried means human labeled.

        meta_df['is_seed']       = False
        meta_df['is_std_val']    = False 
        meta_df['is_std_train']  = False
        meta_df['is_hyp_val']    = False
        meta_df['is_removed']    = False 
        meta_df['removed_in_round'] = -1
        meta_df['is_calib']      = False  # part of calibrtaion dataset or not.
        return meta_df

    
    def get_current_training_data(self,include_auto_labeled=False, include_pseudo_labeled=False):
        train_idcs = self.get_current_train_idcs(include_auto_labeled, include_pseudo_labeled)
        lbld_pts = np.array(train_idcs)
        
        #Y_ = []
        #for i in lbld_pts:
        #    lbl = self.meta_df.at[i,'label']
        #    assert lbl is not None
        #    Y_.append(lbl)
        #Y_ = np.array(Y_)
        return self.ds.get_subset(lbld_pts),train_idcs
    
    def query_training_points(self,n_b,method='random'):
        if(method=='random'):
            train_pts_idcs = self.select_k_unlbld_train_pts_randomly(n_b)
            for idx in train_pts_idcs:
                self.meta_df.at[idx,'is_labeled']= True
                self.meta_df.at[idx,'is_queried']= True
            return train_pts_idcs
        else:
            return [] 
    
    def select_k_unlbld_train_pts_randomly(self,k=10):
        unlbld_pts = self.get_current_unlabeled_train_idcs()
        k = min(k,len(unlbld_pts))

        if(k>0):
            ps = np.random.get_state()
            np.random.seed(42) # use same fixed seed for this.
            z = np.random.choice(unlbld_pts,k,replace=False)
            np.random.set_state(ps)

            return z 
        else:
            return []
    
    def select_seed_train_points(self,k=100,method='randomly'):
        seed_idcs = [] 
        if(method=='randomly'):
            seed_idcs = self.select_k_unlbld_train_pts_randomly(k)
        
        elif(method=='class_balanced_random'):
            lbl_idx_map_std_train = self.get_std_train_set_idcs_class_map()
            classes = lbl_idx_map_std_train.keys()
            C = len(classes)
            for c in classes:
                lst_idcs = np.random.choice(lbl_idx_map_std_train[c],k//C,replace=False)
                seed_idcs.extend(lst_idcs)
        else:
            self.logger.error(f'Unsupported query method {method}')

        for idx in seed_idcs:
            self.meta_df.at[idx,'is_seed']= True  
            self.meta_df.at[idx,'is_queried']= True
            self.meta_df.at[idx,'is_labeled']= True

        return seed_idcs
    
    def get_std_train_set_idcs_class_map(self):
        if(self.lbl_idx_map_std_train is None):
            df_tmp = self.meta_df.query("is_std_train==True & is_removed==False")[['idx','label']]
            self.lbl_idx_map_std_train = df_tmp.groupby('label')['idx'].apply(list).to_dict()
        
        return self.lbl_idx_map_std_train

    def get_std_train_set_idcs(self):
        return self.meta_df.query("is_std_train==True & is_removed==False")['idx'].tolist() 

    def get_seed_train_idcs(self):
        return self.meta_df.query("is_std_train==True & is_removed==False & is_queried==True & is_seed==True")['idx'].tolist()


    def get_current_unlabeled_train_idcs(self, include_pseudo_labeled=True):
        if(include_pseudo_labeled):
            return self.meta_df.query("is_std_train==True & is_removed==False & is_queried==False & is_auto_labeled==False")['idx'].tolist()
        else:
            return self.meta_df.query("is_std_val==True & is_removed==False & is_queried==False & is_auto_labeled==False & is_pseudo_labeled==False")['idx'].tolist()

    def get_current_train_count(self,include_auto_labeled=False, include_pseudo_labeled=False):
        if(include_pseudo_labeled):
            return self.meta_df.query("is_std_train==True & is_removed==False & is_queried==True | is_pseudo_labeled==True")['idx'].count()
        
        elif(include_auto_labeled):
            return self.meta_df.query("is_std_train==True & is_removed==False & is_queried==True | is_auto_labeled==True")['idx'].count()
        else:
            return self.meta_df.query("is_std_train==True & is_removed==False & is_queried==True")['idx'].count()
    
    def get_current_train_idcs(self,include_auto_labeled=False, include_pseudo_labeled=False):

        if(include_pseudo_labeled):
            return self.meta_df.query("is_std_train==True & is_removed==False & is_queried==True | is_pseudo_labeled==True")['idx'].tolist()
        
        elif(include_auto_labeled):
            return self.meta_df.query("is_std_train==True & is_removed==False & is_queried==True | is_auto_labeled==True")['idx'].tolist()
        else:
            return self.meta_df.query("is_std_train==True & is_removed==False & is_queried==True ")['idx'].tolist()

    def get_current_unlabeled_idcs(self, include_pseudo_labeled=True):
        if(include_pseudo_labeled):
            return self.meta_df.query("is_std_train==True & is_labeled==False")['idx'].tolist()
        else:
            return self.meta_df.query("is_std_train==True & is_labeled==False & is_pseudo_labeled==False")['idx'].tolist()
    
    def get_current_unlabeled_count(self, include_pseudo_labeled=True ):
        return len(self.get_current_unlabeled_idcs(include_pseudo_labeled=include_pseudo_labeled))
    
    def get_subset_dataset(self,idcs):
        return self.ds.get_subset(idcs)

    def mark_queried(self,pts_idcs,round_id=0):
        #list_idx_lbl =self.human_labeling_helper.query_labels_for_batch(pts_idcs)
        for idx in pts_idcs:
            #idx = list_idx_lbl[i][0]
            #lbl = list_idx_lbl[i][1]
            #self.meta_df.at[idx,'true_label']         =  lbl
            self.meta_df.at[idx,'is_labeled']    =  True
            self.meta_df.at[idx,'is_queried']    = True
            self.meta_df.at[idx,'queried_in_round'] = round_id
            self.meta_df.at[idx,'confidence']    =  1.0
        
    def mark_auto_labeled(self,auto_labeled_data,round_id=-1):
        for o in auto_labeled_data:
            idx = o['idx']
            self.meta_df.at[idx,'auto_label'] = o['label']
            self.meta_df.at[idx,'is_auto_labeled'] = True
            self.meta_df.at[idx,'is_labeled'] = True
            self.meta_df.at[idx,'auto_labeled_in_round'] = round_id
            self.meta_df.at[idx,'confidence'] = o['confidence']

    def unmark_auto_labeled(self,idx=None):
        if(idx==None):
            idx = self.meta_df.query('is_auto_labeled==True')['idx']
        for i in idx:
            self.meta_df.at[i,'is_auto_labeled']=False
            self.meta_df.at[i,'is_labeled']=False
            self.meta_df.at[i,'auto_labeled_in_round']=-1
            self.meta_df.at[i,'auto_label'] = -1
            self.meta_df.at[i,'confidence'] = 0
    
    def mark_pseudo_labeled(self,pseudo_labeled_data,round_id=-1):
        for o in pseudo_labeled_data:
            idx = o['idx']
            self.meta_df.at[idx,'pseudo_label'] = o['label']
            self.meta_df.at[idx,'is_pseudo_labeled'] = True
            self.meta_df.at[idx,'is_labeled'] = False
            
            self.meta_df.at[idx,'confidence'] = o['confidence']

    def unmark_pseudo_labeled(self,idx=None):
        if(idx==None):
            idx = self.meta_df.query('is_pseudo_labeled==True')['idx']
        for i in idx:
            self.meta_df.at[i,'is_pseudo_labeled']=False
            #self.meta_df.at[i,'is_labeled']=False
            #self.meta_df.at[i,'pseudo_labeled_in_round']=-1
            self.meta_df.at[i,'pseudo_label'] = -1
            self.meta_df.at[i,'confidence'] = 0
    
    def select_k_unlbld_val_pts_randomly(self,k=10):

        unlbld_pts = self.get_current_unlabeled_val_idcs()
        k = min(k,len(unlbld_pts))
        if(k>0):
            ps = np.random.get_state()
            np.random.seed(42) # use same fixed seed for this.
            z= np.random.choice(unlbld_pts,k,replace=False)
            np.random.set_state(ps)
            
            return z 
        else:
            return []
    
    # pseudo labeled are not considered labeled.
    
    def get_current_unlabeled_val_idcs(self,include_pseudo_labeled=True):
        if(include_pseudo_labeled):
            return self.meta_df.query("is_std_val==True & is_removed==False & is_queried==False")['idx'].tolist()
        else:
            return self.meta_df.query("is_std_val==True & is_removed==False & is_queried==False & is_pseudo_labeled==False")['idx'].tolist()
    
    def get_current_validation_count(self):
        return self.meta_df.query("is_std_val==True & is_removed==False & is_labeled==True")['idx'].count()

    def get_current_validation_idcs(self):
        return self.meta_df.query("is_std_val==True & is_removed==False & is_labeled==True")['idx'].tolist()

    def get_current_unlabeled_hyp_idcs(self):
        return self.meta_df.query("is_hyp_val==True & is_removed==False & is_labeled==False")['idx'].tolist()

    def query_validation_points(self,n_v,method='random'):
        if(method=='random'):
            val_pts_idcs = self.query_k_unlbld_val_pts_randomly(n_v)
            for idx in val_pts_idcs:
                self.meta_df.at[idx,'is_labeled']= True
                self.meta_df.at[idx,'is_queried']= True
            return val_pts_idcs
        else:
            return [] 
                
    def get_current_validation_data(self):
        val_idcs = self.get_current_validation_idcs()
        lbld_pts = np.array(val_idcs)
        #Y_ = []
        #for i in lbld_pts:
        #    lbl = self.meta_df.at[i,'true_label']
        #    assert lbl is not None
        #    Y_.append(lbl)
        #Y_ = np.array(Y_)
        return self.ds.get_subset(lbld_pts),val_idcs
    
    def query_k_unlbld_val_pts_randomly(self,k=10):
        unlbld_pts = self.get_current_unlabeled_val_idcs()
        k = min(k,len(unlbld_pts))
        if(k>0):
            return np.random.choice(unlbld_pts,k,replace=False)
        else:
            return []

    def remove_validation_points(self,ids_to_remove,round_id=-1):
        for _id in ids_to_remove:
            self.meta_df.at[_id,'is_removed'] = True
            self.meta_df.at[_id,'removed_in_round']=round_id
    
    def num_labeled_pts(self):
         return self.meta_df.query("is_labeled==True")['idx'].count()

    def clear_val_marks(self):
        for idx in self.idcs_std_val:
            self.meta_df.at[idx,'is_queried'] = False 

    def clear_train_marks(self):
        for idx in self.idcs_std_train:
            self.meta_df.at[idx,'is_queried'] = False 

    def get_auto_labeling_counts(self):

        if(self.conf.eval=='hyp'):
            df_= self.meta_df.query("is_auto_labeled == True & is_hyp_val == True")[['auto_label','true_label']]
            n  = len(self.meta_df.query("is_hyp_val == True")['idx']) 
        else:
            df_= self.meta_df.query("is_auto_labeled == True & is_std_train == True")[['auto_label','true_label']]
            n  = len(self.meta_df.query("is_std_train == True")['idx']) 
        
        y_hat = df_['auto_label'].tolist()
        y_true = df_['true_label'].tolist()
        if(len(y_hat)==0):
            auto_lbl_acc = None # nan case.
        else:
            auto_lbl_acc= accuracy_score(y_hat,y_true)
        auto_lbl_cov = len(y_hat)/n

        if(len(y_hat)==0):
            auto_lbl_acc = None # nan case.

        out = {'auto_labeled_acc':auto_lbl_acc,'coverage_1':auto_lbl_cov,'coverage_2':0}
        return out
    

    def get_pseudo_labeling_counts(self):

        if(self.conf.eval=='hyp'):
            df_= self.meta_df.query("is_pseudo_labeled == True & is_hyp_val == True")[['pseudo_label','true_label']]
            n  = len(self.meta_df.query("is_hyp_val == True")['idx']) 
        else:
            df_= self.meta_df.query("is_pseudo_labeled == True & is_std_train == True")[['pseudo_label','true_label']]
            n  = len(self.meta_df.query("is_std_train == True")['idx']) 
        
        y_hat = df_['pseudo_label'].tolist()
        y_true = df_['true_label'].tolist()
        if(len(y_hat)==0):
            auto_lbl_acc = None # nan case.
        else:
            auto_lbl_acc= accuracy_score(y_hat,y_true)
        auto_lbl_cov = len(y_hat)/n

        if(len(y_hat)==0):
            auto_lbl_acc = None # nan case.

        out = {'pseudo_labeled_acc':auto_lbl_acc,'coverage_1':auto_lbl_cov,'coverage_2':0}
        return out
    
    def select_calib_val_points(self,calib_frac=0.5):
        val_idcs =  self.get_current_validation_idcs() 
        self.logger.info(f'Selecting {calib_frac} fraction of calibration val points out of total :{len(val_idcs)}')
        
        k = min(int(calib_frac*len(val_idcs)), len(val_idcs))

        self.meta_df['is_calib'] = False 

        if(k>0):
            calib_val_idcs = np.random.choice(val_idcs,k,replace=False)
            for idx in calib_val_idcs:
                self.meta_df.at[idx,'is_calib']=True 
            return calib_val_idcs
        else:
            return []
        
    def get_cur_calib_val_ds(self):
        calib_idcs = self.meta_df.query("is_std_val==True & is_removed==False & is_labeled==True & is_calib==True")['idx'].tolist()
        #print(calib_idcs)
        calib_ds = None 
        if(len(calib_idcs)>0):
            calib_ds = self.get_subset_dataset(calib_idcs)

        self.logger.info(f' Cur calib ds size : {len(calib_idcs)}')

        return calib_ds, calib_idcs

    def get_cur_non_calib_val_ds(self):
        no_calib_idcs = self.meta_df.query("is_std_val==True & is_removed==False & is_labeled==True & is_calib==False")['idx'].tolist()
        #no_calib_ds = None 
        #if(len(no_calib_idcs)>0):
        no_calib_ds = self.get_subset_dataset(no_calib_idcs)

        return no_calib_ds , no_calib_idcs


