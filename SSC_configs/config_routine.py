import os
import yaml
import sys

# Append root directory to system path for imports
repo_path, _ = os.path.split(os.path.realpath(__file__))
repo_path, _ = os.path.split(repo_path)
sys.path.append(repo_path)

from LMSCNet.common.io_tools import _create_directory

config_dict = {}

output_root = ''
output_folder = 'routines'
output_filename = 'LMSCNet.yaml'
out_path = os.path.join(output_root, output_folder, output_filename)

# -------------------------------------------------------------
config_dict['DATALOADER'] = {}
config_dict['DATALOADER']['NUM_WORKERS'] = 4
# -------------------------------------------------------------

# -------------------------------------------------------------
config_dict['DATASET'] = {}
config_dict['DATASET']['TYPE'] = 'SemanticKITTI'  # SemanticKITTI, other datasets might be added...
config_dict['DATASET']['MODALITIES'] = {}
# More modalities might be added
config_dict['DATASET']['MODALITIES']['3D_LABEL'] = True
config_dict['DATASET']['MODALITIES']['3D_OCCUPANCY'] = True
config_dict['DATASET']['MODALITIES']['3D_OCCLUDED'] = True
config_dict['DATASET']['ROOT_DIR'] = '/datasets_local/datasets_lroldaoj/semantic_kitti_v1.0/'
config_dict['DATASET']['AUGMENTATION'] = {}
config_dict['DATASET']['AUGMENTATION']['FLIPS'] = True  # More data augmentation can be added in dataloader
# -------------------------------------------------------------

# -------------------------------------------------------------
config_dict['MODEL'] = {}
config_dict['MODEL']['TYPE'] = 'LMSCNet'  # [LMSCNet, LMSCNet_SS, SSCNet, SSCNet_full]
# -------------------------------------------------------------

# -------------------------------------------------------------
config_dict['OPTIMIZER'] = {}
config_dict['OPTIMIZER']['BASE_LR'] = 0.001
config_dict['OPTIMIZER']['TYPE'] = 'Adam'  # [SGD, Adam]
# For SGD Optimizer
config_dict['OPTIMIZER']['MOMENTUM'] = 'NA'
config_dict['OPTIMIZER']['WEIGHT_DECAY'] = 'NA'
# For Adam Optimizer
config_dict['OPTIMIZER']['BETA1'] = 0.9
config_dict['OPTIMIZER']['BETA2'] = 0.999
# -------------------------------------------------------------

# -------------------------------------------------------------
config_dict['OUTPUT'] = {}
config_dict['OUTPUT']['OUT_ROOT'] = '../SSC_out/'
# -------------------------------------------------------------

# -------------------------------------------------------------
config_dict['SCHEDULER'] = {}
config_dict['SCHEDULER']['TYPE'] = 'power_iteration'  # ['constant', 'power_iteration']
config_dict['SCHEDULER']['FREQUENCY'] = 'epoch'
config_dict['SCHEDULER']['LR_POWER'] = 0.98           # ['NA', 0.98]
# -------------------------------------------------------------

# -------------------------------------------------------------
config_dict['STATUS'] = {}
config_dict['STATUS']['RESUME'] = False
# -------------------------------------------------------------

# -------------------------------------------------------------
config_dict['TRAIN'] = {}
config_dict['TRAIN']['BATCH_SIZE'] = 4
config_dict['TRAIN']['CHECKPOINT_PERIOD'] = 15
config_dict['TRAIN']['EPOCHS'] = 80
config_dict['TRAIN']['SUMMARY_PERIOD'] = 50
# -------------------------------------------------------------

# -------------------------------------------------------------
config_dict['VAL'] = {}
config_dict['VAL']['BATCH_SIZE'] = 8
config_dict['VAL']['SUMMARY_PERIOD'] = 20
# -------------------------------------------------------------

_create_directory(os.path.dirname(out_path))
yaml.dump(config_dict, open(out_path, 'w'))

print('Config routine file {} saved...'.format(out_path))
