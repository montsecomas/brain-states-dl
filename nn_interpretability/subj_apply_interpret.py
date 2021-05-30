from captum.attr import IntegratedGradients, ShapleyValueSampling, KernelShap, Lime
import matplotlib.pyplot as plt
import seaborn as sns
from utils.file_utils import load_model
import numpy as np
from nn_classification.data_loaders import SingleSubjectNNData
from nn_interpretability.display_module import importances_results_df, ig_results_loc_scatter_maps, ig_results_loc_surf_maps
import os.path as osp


def compute_attributions(method, subject, freq_name, model, input_tensor, heatmaps_path, electrodes_pos,
                         silent_chan, create_histograms=True):
    if method == 'IntegratedGradients':
        interpreter = IntegratedGradients(model)
    elif method == 'ShapleyValueSampling':
        interpreter = ShapleyValueSampling(model)
    elif method == 'KernelShap':
        interpreter = KernelShap(model)
    elif method == 'Lime':
        interpreter = Lime(model)

    min_imp = np.inf
    max_imp = -np.inf
    attrs = []
    for target_state in [0, 1, 2]:
        attr = interpreter.attribute(input_tensor, target=target_state)
        attr = attr.detach().numpy()
        attrs.append(attr)
        min_imp = attr.min() if attr.min() < min_imp else min_imp
        max_imp = attr.max() if attr.max() > max_imp else max_imp

    # FEATURE IMPORTANCE HEATMAPS FOR EACH LABEL
    if create_histograms:
        plt.rcParams.update({'font.size': 8})
        for target_state in [0, 1, 2]:
            attr = attrs[target_state]
            abs_attr = np.abs(attr)
            sample_tot = abs_attr.sum(axis=1)
            attr_perc = abs_attr/sample_tot.reshape(-1, 1)
            fig = plt.figure(figsize=(15, 9))
            fig.tight_layout()
            fig.suptitle(f'Features importances on validation set examples using {method}\n'
                         f'Subject {subject} - Motivation {target_state} - Frequency {freq_name}')
            for i in range(1, attr.shape[1]+1):
                plt.subplot(8, 9, i)
                plt.hist(attr_perc[:, i - 1], bins=20)
                plt.title(f'{electrodes_pos[i - 1, 5]}')
            fig.tight_layout(pad=1.0)

            plt.savefig(osp.join(heatmaps_path, f'feature_importances_{method}_s{subject}_t{target_state}_{freq_name}.png'))
            plt.clf()

    return attrs


def run_attributions_bloc(cfg, ckpt_paths, draw_histograms, save_tables, draw_dots_map, draw_surf_map,
                          method='IntegratedGradients', nn_use_silent_channels=True):

    freq_ids = dict({'alpha': 0, 'beta': 1, 'gamma': 2})
    heatmaps_path = osp.join(cfg['outputs_path'], cfg['interpret_figures'], cfg['interpret_heatmaps'])

    for subject in cfg['healthy_subjects']:
        # subject = 25
        print('------------------------------------\nSubject', subject,
              '\n------------------------------------')

        # ---- LOAD TRAINED MODEL ----
        if nn_use_silent_channels:
            subject_path = f'exp_logs_subject_fixsilent/subject-{subject}/'
        else:
            subject_path = f'exp_logs_subject_nosilent/subject-{subject}/'

        # ---- LOAD TRAIN VAL DATASET ----
        subject_data = SingleSubjectNNData(subject=subject, classifier='mlp', cfg=cfg,
                                           read_silent_channels=True, force_read_split=True)

        for FREQ in freq_ids.keys():
            # FREQ = 'gamma'
            model_ckpt = f'ss-{FREQ}-mlp-pow-nosilent' if nn_use_silent_channels else f'ss-{FREQ}-mlp-pow-nosilent'
            ckpt_path = ckpt_paths[subject][model_ckpt]
            model = load_model(osp.join(subject_path, ckpt_path), model='mlp')

            freq_id = freq_ids[FREQ]
            train_loader, val_loader = subject_data.mlp_ds_loaders(freq=freq_id)  # 0 alpha, 1 beta, 2 gamma

            batch = next(iter(val_loader))
            inputs, targets = batch
            silent_chan = np.load(osp.join(cfg['data_path'], cfg['healthy_dir'], f'res_subject_{subject}',
                                           f'silent-channels-{subject}.npy'))

            # ---- MAKE PREDICTIONS ----
            test_input_tensor = inputs.float().clone()
            out_probs = model(test_input_tensor).detach().numpy()
            out_classes = np.argmax(out_probs, axis=1)

            print(f"Validation Accuracy freq {FREQ}:", sum(out_classes == targets.numpy()) / len(targets))

            # ---- INTERPRETER ----
            electrodes_pos = np.load(osp.join(cfg['outputs_path'], cfg['electrodes_map'], 'xy_coord.npy'))
            test_input_tensor.requires_grad_()

            # method = 'ShapleyValueSampling'
            attrs = compute_attributions(method=method,
                                         subject=subject,
                                         freq_name=FREQ,
                                         model=model,
                                         input_tensor=test_input_tensor,
                                         heatmaps_path=heatmaps_path,
                                         electrodes_pos=electrodes_pos,
                                         create_histograms=draw_histograms,
                                         silent_chan=silent_chan)

            target_dfs = importances_results_df(subject, FREQ, cfg, attrs, electrodes_pos, silent_chan, save_tables)

            if draw_dots_map:
                ig_results_loc_scatter_maps(subject, FREQ, cfg, target_dfs, electrodes_pos, method=method,
                                            silent_chan=silent_chan)

            if draw_surf_map:
                ig_results_loc_surf_maps(subject, FREQ, cfg, target_dfs, electrodes_pos, method=method,
                                         silent_chan=silent_chan)

