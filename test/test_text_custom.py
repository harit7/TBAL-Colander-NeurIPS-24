import sys 
sys.path.append('../')
from src.utils.run_lib import * 

from src.core.passive_learning import *
from src.core.auto_labeling import *
from src.utils.logging_utils import * 
from src.datasets import dataset_factory  
from src.datasets.dataset_utils import * 
from src.utils.counting_utils import *  
from src.utils.common_utils import * 
from src.utils.vis_utils import *
import copy 
import random

from omegaconf import OmegaConf
#from calibration_utils import * 
from  src.datasets.data_manager import * 

from src.calibration.calibration_utils import * 

# configuration
conf_dir = '../configs/test/'

act_learn_conf_file = '{}/multi_nli_bert_base_conf.yaml'.format(conf_dir)

#conf = load_yaml_config(pas_learn_conf_file)
conf = OmegaConf.load(act_learn_conf_file)

conf.data_conf.compute_emb = False 

#run_conf(conf)

set_seed(conf['random_seed'])
dm = DataManager(conf,logger)


logger   = get_logger('../temp/logs/multi_nli_act_learning.log',stdout_redirect=True,level=logging.DEBUG)

act_lbl = ActiveLabeling(conf,dm,logger)

act_lbl.init()

# breakpoint()
lst_epoch_out = act_lbl.run_al_loop()


logger   = get_logger('../temp/logs/multi_nli_passive_learning.log',stdout_redirect=True,level=logging.DEBUG)
pl = PassiveLearning(conf,dm,logger)

out = pl.run()

w = pl.cur_clf.get_weights()
print(torch.norm(w))
test_err = get_test_error(pl.cur_clf,dm.ds_std_test,conf['inference_conf'])
print(test_err)
