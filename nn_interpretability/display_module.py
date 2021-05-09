from captum.attr import IntegratedGradients
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os.path as osp
import pandas as pd


def compute_integrated_gradients(subject, model, input_tensor, heatmaps_path, create_heatmap=True):
    ig = IntegratedGradients(model)

    min_imp = np.inf
    max_imp = -np.inf
    attrs = []
    deltas = []
    for target_state in [0, 1, 2]:
        attr, delta = ig.attribute(input_tensor, target=target_state, return_convergence_delta=True)
        attr = attr.detach().numpy()
        attrs.append(attr)
        deltas.append(delta)
        min_imp = attr.min() if attr.min() < min_imp else min_imp
        max_imp = attr.max() if attr.max() > max_imp else max_imp

    # FEATURE IMPORTANCE HEATMAPS FOR EACH LABEL
    if create_heatmap:
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

    return attrs


def ig_results_df(subject, cfg, attrs, electrodes_pos, save_tables=True):
    mean_attrs = pd.DataFrame(np.array([np.mean(attr, axis=0) for attr in attrs]).T)
    mean_attrs.columns = ['raw_mot_0', 'raw_mot_1', 'raw_mot_2']
    mean_attrs['electrode_id'] = (mean_attrs.index + 1).astype(int)

    electrodes_t_names = pd.DataFrame(np.array([electrodes_pos[:, 0], electrodes_pos[:, -1]]).T)
    electrodes_t_names.columns = ['electrode_id', 'electrode_name']
    electrodes_t_names['electrode_id'] = electrodes_t_names['electrode_id'].astype(int)

    mean_attrs_names = mean_attrs.merge(electrodes_t_names, on='electrode_id')

    target_dfs = []
    for target_state in [0, 1, 2]:
        mean_attrs_names[f'abs_mot_{target_state}'] = np.abs(mean_attrs_names[f'raw_mot_{target_state}'])
        mean_attrs_names[f'perc_mot_{target_state}'] = \
            mean_attrs_names[f'abs_mot_{target_state}'] / mean_attrs_names[f'abs_mot_{target_state}'].sum()

        target_dfs.append(mean_attrs_names[['electrode_id', 'electrode_name',
                                            f'raw_mot_{target_state}',
                                            f'abs_mot_{target_state}',
                                            f'perc_mot_{target_state}']].
                          sort_values(f'perc_mot_{target_state}', ascending=False))

        if save_tables:
            target_dfs[target_state].to_csv(osp.join(cfg['outputs_path'], cfg['interpret_imp_mlp'],
                                                     f'{subject}_mlp_importances_motiv{target_state}.csv'),
                                            index=False)

    return target_dfs


def ig_results_loc_maps(subject, cfg, target_dfs, electrodes_pos, save_tables=True):
    for target_state in [0, 1, 2]:
        imp = target_dfs[target_state][['electrode_name', f'perc_mot_{target_state}']]
        electr_coord = pd.DataFrame(np.array([electrodes_pos[:, -1],
                                              electrodes_pos[:, 1].astype(float),
                                              electrodes_pos[:, 2].astype(float)]).T)
        electr_coord.columns = ['electrode_name', 'x', 'y']
        imp_coord = imp.merge(electr_coord, on='electrode_name')

        xs = np.array(electr_coord.x).astype(float)
        ys = np.array(electr_coord.y).astype(float)

        fig, ax = plt.subplots()
        im = ax.scatter(xs, ys, s=30, c=imp_coord[f'perc_mot_{target_state}'] * 100)
        fig.colorbar(im, ax=ax)

        ax.set_title(f'MLP: F.Importance with Integrated Gradients \nSubject {subject} - Target state {target_state}')
        fig.savefig(osp.join(cfg['outputs_path'], cfg['interpret_figures'], cfg['interpret_loc'],
                             f'{subject}_electrode_loc_motiv{target_state}.png'))
        plt.close('all')
