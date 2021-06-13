import numpy as np
import os.path as osp
import re
import matplotlib.pyplot as plt
from utils.file_utils import load_cfg

coordinates = 'xy'
do_surface = False
cfg = load_cfg()
sufix_str = '_pd' if cfg['run_pd'] else ''
# sufix_str = ''
coords_file = 'GSN128.sfp' if cfg['run_pd'] else 'ActiCap64_LM.lay'
# coords_file = 'ActiCap64_LM.lay'

filename = osp.join(cfg['outputs_path'], cfg[f'electrodes_map{sufix_str}'], coords_file)
with open(filename) as f:
    content = f.readlines()

content = [x.strip() for x in content]
str_content = np.asarray([re.split('\t', line.replace(' ', '')) for line in content])

np.save(osp.join(cfg['outputs_path'], cfg[f'electrodes_map{sufix_str}'], 'xy_coord.npy'), str_content)
xs = str_content[:, 1].astype(float)
ys = str_content[:, 2].astype(float)
tags = np.array([i[3:] for i in str_content[:, 0]]) if cfg['run_pd'] else str_content[:, -1]

plt.rcParams.update({'font.size': 8})
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(xs, ys, s=30)

for i, txt in enumerate(tags):
    ax.annotate(txt, (xs[i], ys[i]))

ax.set_title('Electrodes locations')

fig.savefig(osp.join(cfg['outputs_path'], cfg['electrodes_map'], 'electrodes_loc.png'))
