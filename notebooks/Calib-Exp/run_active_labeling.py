import sys 
sys.path.append('../../../')
sys.path.append('../../')
sys.path.append('../')

from utils.logging_utils import * 
from utils.counts import *  
from utils.common_utils import * 
from utils.vis_utils import *

from core.run_lib import *
from core.conf_defaults import set_defaults

import utils


active_lbl_config_file = '../../configs/arxiv-configs/Cifar10-SmallNet/active_labeling_cifar10_small_net_histogram.yaml'

logger = get_logger('../../temp/logs/AL.log',True,level=logging.INFO)

conf = load_yaml_config(active_lbl_config_file)
set_defaults(conf)


n_q = 20000
N = 50000
val_frac = 0.2
eps = 0.10

val_frac_auto_lbl = 1.0 # 10k

N_pool = int(N*(1-val_frac))
val_num_auto_lbl = int(N*val_frac*val_frac_auto_lbl)

train_conf = conf['training_conf']
conf['max_query'] = n_q 
conf['auto_lbl_conf']['N_V_max'] = val_num_auto_lbl

out = run_active_labeling_conf(conf,logger,return_per_epoch_out=True)   