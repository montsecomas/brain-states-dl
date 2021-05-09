import os
import os.path as osp
from utils.file_utils import load_cfg

cfg = load_cfg()

PATHS = [cfg['data'],
         osp.join(cfg['data'], cfg['pd_dir']),
         osp.join(cfg['data'], cfg['healthy_dir']),
         osp.join(cfg['data'], cfg['healthy_dir'], cfg['splits_path']),
         cfg['outputs_path'],
         osp.join(cfg['outputs_path'], cfg['interpret_figures']),
         osp.join(cfg['outputs_path'], cfg['interpret_figures'], cfg['interpret_heatmaps'])]

for path in PATHS:
    if not osp.exists(path):
        os.mkdir(cfg['data'])
