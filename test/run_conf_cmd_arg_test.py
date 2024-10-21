import sys

sys.path.append('../')
sys.path.append('../../')


from omegaconf import OmegaConf
from src.utils.run_lib import *

# Check if command line argument has one argument
if len(sys.argv) != 2:
    print("Usage: python run_conf_cmd_arg_test.py <path_to_run_config>")
    exit()
# Read command line argument and assign it to conf_file_path
conf_file_path = sys.argv[1]
conf = OmegaConf.load(conf_file_path)

run_conf(conf,overwrite=True,stdout=True)