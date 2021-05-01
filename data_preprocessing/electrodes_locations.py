import numpy as np
import os.path as osp
import re


filename = osp.join('data', 'healthy_sb', 'electrodes_loc', 'easycap-M10.txt')
with open(filename) as f:
    content = f.readlines()[1:]
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content]
str_content = np.asarray([re.split('\t', line.replace(' ', '')) for line in content])

