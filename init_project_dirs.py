import os
import os.path as osp
from utils.file_utils import load_cfg

cfg = load_cfg()

PATHS = [cfg['data_path'],
         osp.join(cfg['data_path'], cfg['pd_dir']),
         osp.join(cfg['data_path'], cfg['healthy_dir']),
         cfg['outputs_path'],
         osp.join(cfg['outputs_path'], cfg['splits_path']),
         osp.join(cfg['outputs_path'], cfg['interpret_figures']),
         osp.join(cfg['outputs_path'], cfg['interpret_figures'], cfg['interpret_heatmaps']),
         osp.join(cfg['outputs_path'], cfg['interpret_figures'], cfg['interpret_loc']),
         osp.join(cfg['outputs_path'], cfg['interpret_figures_pd']),
         osp.join(cfg['outputs_path'], cfg['interpret_figures_pd'], cfg['interpret_heatmaps']),
         osp.join(cfg['outputs_path'], cfg['interpret_figures_pd'], cfg['interpret_loc']),
         osp.join(cfg['outputs_path'], cfg['interpret_imp_mlp']),
         osp.join(cfg['outputs_path'], cfg['interpret_imp_mlp_pd'])]

for path in PATHS:
    if not osp.exists(path):
        os.mkdir(path)
