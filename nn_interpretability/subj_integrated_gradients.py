import sys
sys.path.append('/Users/mcomastu/TFM/brain-states-dl') # TODO: remove this first two lines

from utils.file_utils import load_cfg, load_model
import numpy as np
from nn_classification.data_loaders import SingleSubjectNNData
from captum.attr import IntegratedGradients
import seaborn as sns
import matplotlib.pylab as plt
import os.path as osp


cfg = load_cfg()
ckpt_paths = load_cfg('nn_interpretability/ckpt_paths.yml')
heatmaps_path = osp.join(cfg['outputs_path'], cfg['interpret_figures'], cfg['interpret_heatmaps'])
FREQ = 'gamma'

for subject in cfg['healthy_subjects']:
    # subject = 25
    print('------------------------------------\nSubject', subject,
          '\n------------------------------------')

    model_ckpt = 'ss-gamma-mlp-pow' if FREQ == 'gamma' else ''
    freq_id = 2 if FREQ == 'gamma' else 0

    # ---- LOAD TRAINED MODEL ----
    subject_path = f'exp_logs_subject/subject-{subject}/'
    ckpt_path = ckpt_paths[subject][model_ckpt]
    model = load_model(osp.join(subject_path, ckpt_path), model='mlp')

    # ---- LOAD TRAIN VAL DATASET ----
    subject_data = SingleSubjectNNData(subject=subject, classifier='mlp', cfg=cfg, force_read_split=True)
    train_loader, val_loader = subject_data.mlp_ds_loaders(freq=freq_id)  # 0 alpha, 1 beta, 2 gamma

    batch = next(iter(val_loader))
    inputs, targets = batch

    # ---- MAKE PREDICTIONS ----
    test_input_tensor = inputs.float().clone()
    out_probs = model(test_input_tensor).detach().numpy()
    out_classes = np.argmax(out_probs, axis=1)

    print("Validation Accuracy:", sum(out_classes == targets.numpy()) / len(targets))

    # ---- INTERPRETER ----
    ig = IntegratedGradients(model)

    test_input_tensor.requires_grad_()

    min_imp = np.inf
    max_imp = -np.inf
    attrs = []
    deltas = []
    for target_state in [0, 1, 2]:
        attr, delta = ig.attribute(test_input_tensor, target=target_state, return_convergence_delta=True)
        attr = attr.detach().numpy()
        attrs.append(attr)
        deltas.append(delta)
        min_imp = attr.min() if attr.min() < min_imp else min_imp
        max_imp = attr.max() if attr.max() > max_imp else max_imp

    # FEATURE IMPORTANCE HEATMAPS FOR EACH LABEL
    for target_state in [0, 1, 2]:
        attr = attrs[target_state]
        ax = plt.axes()
        sns.heatmap(attr, ax=ax, vmin=min_imp, vmax=max_imp, cmap='vlag')
        ax.set_title(f'Feature importances for motivational state {target_state}')
        plt.savefig(osp.join(heatmaps_path, f'feature_importances_s{subject}_t{target_state}.png'))
        plt.clf()

        attr = attrs[target_state]
        ax = plt.axes()
        sns.heatmap(np.mean(attr, axis=0).reshape(1, -1), ax=ax, vmin=min_imp, vmax=max_imp, cmap='vlag')
        ax.set_title(f'AVERAGE Feature importances for motivational state {target_state}')
        plt.savefig(osp.join(heatmaps_path, f'feature_importances_s{subject}_t{target_state}_avg.png'))
        plt.clf()

    # visualize_importances(feature_names=np.arange(inputs.shape[1])+1, importances=np.mean(attr, axis=0),
    #                       title=f'Subject {subject} \nAverage Feature Importances for motivational state {target_state}')

    # electrodes_pos = np.load('data/healthy_sb/electrodes_loc/xy_coord.npy')


