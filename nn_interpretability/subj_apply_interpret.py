from captum.attr import IntegratedGradients, ShapleyValueSampling, KernelShap, Lime
import matplotlib.pyplot as plt
import seaborn as sns
from utils.file_utils import load_model
import numpy as np
from nn_classification.data_loaders import SingleSubjectNNData
from nn_interpretability.display_module import importances_results_df_beta, ig_results_loc_scatter_maps, \
    ig_results_loc_surf_maps, input_power_surf
import os.path as osp
import glob


def compute_attributions(method, subject, freq_name, model, input_tensor, heatmaps_path, electrodes_pos,
                         silent_chan, create_histograms=True, session_sufix=''):
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
        ncols = 9
        nrows = np.ceil(attrs[0].shape[1]/ncols).astype(int)
        plt.rcParams.update({'font.size': 8})
        for target_state in [0, 1, 2]:
            attr = attrs[target_state]
            abs_attr = np.abs(attr)
            sample_tot = abs_attr.sum(axis=1)
            attr_perc = abs_attr/sample_tot.reshape(-1, 1)
            if session_sufix == '':
                fig = plt.figure(figsize=(15, 9))
            else:
                fig = plt.figure(figsize=(9, 15))
            fig.tight_layout()
            fig.suptitle(f'Features importances on validation set examples using {method}\n'
                         f'Subject {subject} - Motivation {target_state} - Frequency {freq_name}')
            for i in range(1, attr.shape[1]+1):
                plt.subplot(nrows, ncols, i)
                plt.hist(attr_perc[:, i - 1], bins=20)
                if session_sufix == '':
                    plt.xlim(xmin=0, xmax=0.25)
                    plt.title(f'{electrodes_pos[i - 1, 5]}')
                else:
                    plt.title(f'{electrodes_pos[i - 1, 0]}')
            fig.tight_layout(pad=1.0)

            plt.savefig(osp.join(heatmaps_path, f'feature_importances_{method}_s{subject}{session_sufix}'
                                                f'_t{target_state}_{freq_name}.png'))
            plt.clf()

    return attrs


def run_attributions_bloc(cfg, ckpt_paths, draw_histograms, save_tables, draw_dots_map, draw_surf_map,
                          draw_surf_pow, method='IntegratedGradients'):

    freq_ids = dict({'alpha': 0, 'beta': 1, 'gamma': 2})
    key_sufix = '_pd' if cfg['run_pd'] else ''
    heatmaps_path = osp.join(cfg['outputs_path'], cfg[f'interpret_figures{key_sufix}'], cfg['interpret_heatmaps'])
    if cfg['run_pd']:
        session = '-on' if cfg['model_on_med'] else '-off'
        subjects_list = cfg['pd_subjects']
        inputs_path = cfg['pd_dir']
    else:
        session = ''
        subjects_list = cfg['healthy_subjects']
        inputs_path = cfg['healthy_dir']

    for subject in subjects_list:
        # subject = 25, healthy
        # subject = 55, PD
        print('------------------------------------\nSubject', subject,
              '\n------------------------------------')

        # ---- LOAD TRAINED MODEL ----
        if cfg['run_pd']:
            subject_path = f'{ckpt_paths["mlp-single-pd-subject-logs"]}/subject-{subject}/'
        else:
            subject_path = f'{ckpt_paths["mlp-single-healthy-subject-logs"]}/subject-{subject}/'

        # ---- LOAD TRAIN VAL DATASET ----
        subject_data = SingleSubjectNNData(subject=subject, classifier='mlp', cfg=cfg,
                                           read_silent_channels=True, force_read_split=True)
        for FREQ in freq_ids.keys():
            # FREQ = 'beta'
            mpathc = glob.glob(subject_path + f'freq-{FREQ}-single_subject/MLP{session}*/checkpoints/*.ckpt')[0]
            model = load_model(mpathc, model='mlp')

            freq_id = freq_ids[FREQ]
            train_loader, val_loader = subject_data.mlp_ds_loaders(freq=freq_id)  # 0 alpha, 1 beta, 2 gamma

            batch = next(iter(val_loader))
            inputs, targets = batch
            silent_chan = np.load(osp.join(cfg['data_path'], inputs_path, f'res_subject_{subject}',
                                           f'silent-channels-{subject}.npy'))

            # ---- MAKE PREDICTIONS ----
            test_input_tensor = inputs.float().clone()
            out_probs = model(test_input_tensor).detach().numpy()
            out_classes = np.argmax(out_probs, axis=1)

            print(f"Validation Accuracy freq {FREQ}:", sum(out_classes == targets.numpy()) / len(targets))

            # ---- INTERPRETER ----
            electrodes_pos = np.load(osp.join(cfg['outputs_path'], cfg[f'electrodes_map{key_sufix}'], 'xy_coord.npy'))
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
                                         silent_chan=silent_chan,
                                         session_sufix=session)

            target_dfs = importances_results_df_beta(subject, FREQ, cfg, attrs, electrodes_pos, silent_chan, save_tables,
                                                     session_sufix=session)

            if draw_dots_map and not cfg['run_pd']:
                ig_results_loc_scatter_maps(subject, FREQ, cfg, target_dfs, electrodes_pos, method=method,
                                            silent_chan=silent_chan)

            if draw_surf_map:
                ig_results_loc_surf_maps(subject, FREQ, cfg, target_dfs, electrodes_pos, method=method,
                                         silent_chan=silent_chan, session_sufix=session)

            if draw_surf_pow:
                input_power_surf(subject, FREQ, cfg, test_input_tensor.detach().numpy(), electrodes_pos,
                                 silent_chan=silent_chan, session_sufix=session)

