import sys
sys.path.append('/Users/mcomastu/TFM/brain-states-dl') # TODO: remove this first two lines

from utils.file_utils import load_cfg, load_model
import numpy as np
from nn_classification.data_loaders import SingleSubjectNNData
from nn_interpretability.display_module import compute_integrated_gradients, ig_results_df, ig_results_loc_maps
import os.path as osp
import pandas as pd
import matplotlib.pyplot as plt


def run_integrated_gradients(cfg, ckpt_paths, FREQ, draw_heatmaps, save_tables, draw_elec_map):
    heatmaps_path = osp.join(cfg['outputs_path'], cfg['interpret_figures'], cfg['interpret_heatmaps'])
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
        electrodes_pos = np.load(osp.join(cfg['outputs_path'], cfg['electrodes_map'], 'xy_coord.npy'))
        test_input_tensor.requires_grad_()

        attrs = compute_integrated_gradients(subject=subject,
                                             model=model,
                                             input_tensor=test_input_tensor,
                                             heatmaps_path=heatmaps_path,
                                             create_heatmap=draw_heatmaps)

        target_dfs = ig_results_df(subject, cfg, attrs, electrodes_pos, save_tables)

        if draw_elec_map:
            ig_results_loc_maps(subject, cfg, target_dfs, electrodes_pos, save_tables)


if __name__ == '__main__':
    cfg = load_cfg()
    ckpt_paths = load_cfg('nn_interpretability/ckpt_paths.yml')
    FREQ = 'gamma'
    DO_HEATMAPS = False
    DO_TABLES = False
    DO_LOC_MAP = True

    run_integrated_gradients(cfg, ckpt_paths, FREQ, draw_heatmaps=DO_HEATMAPS, save_tables=DO_TABLES,
                             draw_elec_map=DO_LOC_MAP)
