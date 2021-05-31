import numpy as np
import os.path as osp
import re
import matplotlib.pyplot as plt
from utils.file_utils import load_cfg

coordinates = 'xy'
do_surface = False
cfg = load_cfg()

filename = osp.join(cfg['outputs_path'], cfg['electrodes_map'], 'ActiCap64_LM.lay')
with open(filename) as f:
    content = f.readlines()

content = [x.strip() for x in content]
str_content = np.asarray([re.split('\t', line.replace(' ', '')) for line in content])

np.save(osp.join(cfg['outputs_path'], cfg['electrodes_map'], 'xy_coord.npy'), str_content)
xs = str_content[:, 1].astype(float)
ys = str_content[:, 2].astype(float)
tags = str_content[:, -1]

fig, ax = plt.subplots()
ax.scatter(xs, ys, s=30)

for i, txt in enumerate(tags):
    ax.annotate(txt, (xs[i], ys[i]))

ax.set_title('Electrodes locations')

fig.savefig(osp.join(cfg['outputs_path'], cfg['electrodes_map'], 'electrodes_loc.png'))





