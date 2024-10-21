import sys

sys.path.append('../')


from omegaconf import OmegaConf
from src.utils.run_lib import *

#conf_file_path ="../../../outputs/mnist_post_hoc_calib-exp-runs/C__2/C_1__0.25/calib_conf__scaling_binning/calib_val_frac__0.5/training_conf.learning_rate__0.5/training_conf.num_bins__10/eps__0.05/max_num_train_pts__500/max_num_val_pts__2000/method__active_labeling/query_batch_frac__0.05/seed__0/seed_frac__0.2/run_config.yaml"
#conf_file_path = '../../../outputs/mnist_post_hoc_calib-exp-runs/C__2/C_1__0.25/calib_conf__auto_label_opt_v0/calib_val_frac__0.5/training_conf_g.learning_rate__0.001/training_conf_g.optimizer__adam/training_conf_t.learning_rate__0.0001/eps__0.05/max_num_train_pts__500/max_num_val_pts__2000/method__active_labeling/query_batch_frac__0.05/seed__0/seed_frac__0.2/run_config.yaml'

conf_file_path = "/home/harit/workspace/TBAL-calib/outputs/mnist_lenet_calib-v2/C__2/C_1__0.25/calib_conf__auto_label_opt_v2/calib_val_frac__0.6/features_key__pre_logits/l1__1.0/l2__0.0/l3__1.0/l4__2.0/regularize__False/training_conf_g.learning_rate__0.0001/training_conf_g.max_epochs__500/training_conf_g.optimizer__adam/eps__0.05/max_num_train_pts__250/max_num_val_pts__2500/method__active_labeling/query_batch_frac__0.05/seed__0/seed_frac__0.2/run_config.yaml"
conf = OmegaConf.load(conf_file_path)

run_conf(conf,overwrite=True,stdout=True)
