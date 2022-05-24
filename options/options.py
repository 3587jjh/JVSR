import os
import yaml
from utils.utils import OrderedYaml
Loader, Dumper = OrderedYaml()


def parse(config_path, mode):
    with open(config_path, mode='r') as f:
        opt = yaml.load(f, Loader=Loader)

    opt['mode'] = mode
    opt['dataset']['path'] = os.path.join(opt['dataset']['path'], mode)
    assert opt['dataset']['GT_size'] % opt['scale'] == 0
    opt['dataset']['LR_size'] = opt['dataset']['GT_size'] // opt['scale']
    opt['train']['num_epochs'] = opt['train']['niter'] // opt['dataset']['batch_size']
    return opt
    
