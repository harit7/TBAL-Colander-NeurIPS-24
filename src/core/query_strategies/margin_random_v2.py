import numpy as np 
import torch

class MarginRandomV2():
    def __init__(self,dm,clf,conf,logger):
        self.dm       = dm 
        self.clf      = clf 
        self.conf     = conf 
        self.logger   = logger 
        pass 
    
    def query_points(self,batch_size,inf_out=None):

        C = self.conf.train_pts_query_conf['margin_random_v2_constant']
        
        cur_unlbld_idcs  = self.dm.get_current_unlabeled_idcs(include_pseudo_labeled=False)
        cur_unlbld_ds    = self.dm.get_subset_dataset(cur_unlbld_idcs)

        if(inf_out is None):
            self.logger.debug('running infernce')
            inf_out = self.clf.predict(cur_unlbld_ds, self.conf['inference_conf']) 
        
        confidence_scores = inf_out['probs']
        if type(confidence_scores) == np.ndarray or type(confidence_scores) == list:    
            probs = torch.Tensor(np.array(confidence_scores))
        else:
            probs = confidence_scores
        
        probs_sorted, _ = probs.sort(descending=True)
        U               = probs_sorted[:, 0] - probs_sorted[:,1]
        
        
        idx             = U.sort()[1].numpy()[:min(C*batch_size,len(U))]
        
        selected_idcs   = np.array(cur_unlbld_idcs)[idx]
        selected_idcs   = selected_idcs.astype(int)

        sample_size     = min(batch_size,len(selected_idcs))
        
        

        selected_idcs = np.random.choice(selected_idcs,sample_size,replace=False)
        
        return selected_idcs
