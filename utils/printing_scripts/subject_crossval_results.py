import numpy as np
import os.path as osp
from utils.file_utils import load_cfg
import matplotlib.pyplot as plt

training_aucs = []
validation_aucs = []
training_accs = []
validation_accs = []

cfg = load_cfg()
ckpt_paths = load_cfg('nn_interpretability/ckpt_paths.yml')
if cfg['run_pd']:
    session = '-on' if cfg['model_on_med'] else '-off'
    subjects_list = cfg['pd_subjects']
    inputs_path = cfg['pd_dir']
else:
    session = ''
    subjects_list = cfg['healthy_subjects']
    inputs_path = cfg['healthy_dir']
freq_ids = dict({'alpha': 0, 'beta': 1, 'gamma': 2})


min_max_accs = np.array([
    [[625, 703, 625, 766, 703, 797],[531, 670, 523, 654, 700, 815]],
    [[579, 632, 684, 790, 579, 790],[608, 667, 603, 658, 598, 706]],
    [[406, 484, 531, 672, 750, 844],[400, 485, 531, 585, 762, 823]],
    [[438, 516, 516, 672, 500, 750],[369, 462, 464, 542, 577, 662]],
    [[453, 688, 438, 578, 438, 594],[519, 583, 510, 574, 546, 620]],
    [[625, 672, 641, 719, 797, 906],[600, 670, 631, 700, 746, 854]],
    [[547, 656, 531, 672, 641, 797],[509, 648, 593, 648, 657, 732]],
    [[610, 719, 641, 813, 719, 875],[623, 685, 669, 700, 731, 831]],
    [[313, 422, 344, 563, 500, 719],[269, 362, 315, 469, 500, 677]],
    [[375, 469, 313, 484, 484, 641],[362, 446, 385, 523, 477, 554]],
    [[359, 547, 453, 578, 609, 719],[392, 492, 462, 546, 608, 715]],
])/1000

x_axis = ['alpha', 'alpha', 'beta', 'beta', 'gamma', 'gamma']
cmaps = ['green', 'green', 'blue', 'blue', 'pink', 'pink']

avg_min_max = np.mean(min_max_accs, axis=0)
std_min_max = np.std(min_max_accs, axis=0)

for i, subject in enumerate(subjects_list):
    # i=0
    train_accs = min_max_accs[i, 0, :]
    val_accs = min_max_accs[i, 1, :]
    plt.subplot(111)
    plt.scatter(x_axis, train_accs, c=cmaps)
    plt.ylim((0, 1))
    plt.title(f'Minimum and maximum accuracy for subject {subject} using MLP')
    plt.savefig(osp.join(cfg['outputs_path'], 'acc_ranges',
                         f'acc_{subject}.png'))
    plt.close('all')


