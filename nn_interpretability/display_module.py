import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import os.path as osp
import pandas as pd
from shapely.geometry import Point
import random
from scipy.interpolate import griddata


def importances_results_df(subject, freq_name, cfg, attrs, electrodes_pos, silent_chan, save_tables=False, session_sufix=''):
    mean_attrs = pd.DataFrame(np.array([np.mean(attr, axis=0) for attr in attrs]).T)
    median_attrs = pd.DataFrame(np.array([np.median(attr, axis=0) for attr in attrs]).T)
    mm_attrs = pd.concat([mean_attrs, median_attrs], axis=1)
    mm_attrs.columns = ['raw_mean_mot_0', 'raw_mean_mot_1', 'raw_mean_mot_2',
                        'raw_median_mot_0', 'raw_median_mot_1', 'raw_median_mot_2']
    mm_attrs['electrode_id'] = (mean_attrs.index + 1).astype(int)

    if session_sufix == '':
        electrodes_t_names = pd.DataFrame(np.array([electrodes_pos[:, 0], electrodes_pos[:, -1]]).T)
        electrodes_t_names.columns = ['electrode_id', 'electrode_name']
        electrodes_t_names['electrode_id'] = electrodes_t_names['electrode_id'].astype(int)
        electrodes_t_names['is_silent'] = silent_chan
        mm_attrs_names = mm_attrs.merge(electrodes_t_names, on='electrode_id')
    else:
        mm_attrs_names = mm_attrs.copy()
        mm_attrs_names['electrode_name'] = mm_attrs_names['electrode_id'].astype(str)
        mm_attrs_names['is_silent'] = silent_chan

    target_dfs = []
    for target_state in [0, 1, 2]:
        mm_attrs_names[f'abs_mean_mot_{target_state}'] = np.abs(mm_attrs_names[f'raw_mean_mot_{target_state}'])
        mm_attrs_names[f'abs_median_mot_{target_state}'] = np.abs(mm_attrs_names[f'raw_median_mot_{target_state}'])

        mm_attrs_names[f'perc_mean_mot_{target_state}'] = \
            mm_attrs_names[f'abs_mean_mot_{target_state}'] / mm_attrs_names[f'abs_mean_mot_{target_state}'].sum()
        mm_attrs_names[f'perc_median_mot_{target_state}'] = \
            mm_attrs_names[f'abs_median_mot_{target_state}'] / mm_attrs_names[f'abs_median_mot_{target_state}'].sum()

        target_dfs.append(mm_attrs_names[['electrode_id', 'electrode_name', 'is_silent',
                                          f'raw_mean_mot_{target_state}',
                                          f'raw_median_mot_{target_state}',
                                          f'perc_mean_mot_{target_state}',
                                          f'perc_median_mot_{target_state}']].
                          sort_values(f'perc_mean_mot_{target_state}', ascending=False))

        if save_tables:
            target_dfs[target_state].to_csv(osp.join(cfg['outputs_path'], cfg['interpret_imp_mlp'],
                                                     f'{subject}_mlp_importances_motiv{target_state}-freq_{freq_name}.csv'),
                                            index=False)

    return target_dfs


def importances_results_df_beta(subject, freq_name, cfg, attrs, electrodes_pos, silent_chan, save_tables=False,
                                session_sufix=''):
    perc_attrs = []
    for target_state in [0, 1, 2]:
        attr = np.abs(attrs[target_state])
        attr_perc = attr / attr.sum(axis=1).reshape(-1, 1)
        perc_attrs.append(attr_perc)

    mean_attrs = pd.DataFrame(np.array([np.mean(attr, axis=0) for attr in perc_attrs]).T)
    median_attrs = pd.DataFrame(np.array([np.median(attr, axis=0) for attr in perc_attrs]).T)
    mm_attrs = pd.concat([mean_attrs, median_attrs], axis=1)
    mm_attrs.columns = ['perc_mean_mot_0', 'perc_mean_mot_1', 'perc_mean_mot_2',
                        'perc_median_mot_0', 'perc_median_mot_1', 'perc_median_mot_2']
    mm_attrs['electrode_id'] = (mean_attrs.index + 1).astype(int)

    if session_sufix == '':
        electrodes_t_names = pd.DataFrame(np.array([electrodes_pos[:, 0], electrodes_pos[:, -1]]).T)
        electrodes_t_names.columns = ['electrode_id', 'electrode_name']
        electrodes_t_names['electrode_id'] = electrodes_t_names['electrode_id'].astype(int)
        electrodes_t_names['is_silent'] = silent_chan
        mm_attrs_names = mm_attrs.merge(electrodes_t_names, on='electrode_id')
    else:
        mm_attrs_names = mm_attrs.copy()
        mm_attrs_names['electrode_name'] = mm_attrs_names['electrode_id'].astype(str)
        mm_attrs_names['is_silent'] = silent_chan

    target_dfs = []
    for target_state in [0, 1, 2]:
        target_dfs.append(mm_attrs_names[['electrode_id', 'electrode_name', 'is_silent',
                                          f'perc_mean_mot_{target_state}',
                                          f'perc_median_mot_{target_state}']].
                          sort_values(f'perc_mean_mot_{target_state}', ascending=False))

        if save_tables:
            ses_path = cfg['interpret_imp_mlp'] if session_sufix == '' else cfg['interpret_imp_mlp_pd']
            target_dfs[target_state].to_csv(osp.join(cfg['outputs_path'], ses_path,
                                                     f'{subject}{session_sufix}_mlp_importances_motiv{target_state}'
                                                     f'-freq_{freq_name}.csv'),
                                            index=False)

    return target_dfs


def random_points_within(poly, num_points):
    min_x, min_y, max_x, max_y = poly.bounds

    points = []

    while len(points) < num_points:
        random_point = Point([random.uniform(min_x, max_x), random.uniform(min_y, max_y)])
        if random_point.within(poly):
            points.append(random_point)

    return points


def eeg_convex_hull(xs, ys, do_eeg_plot=True):
    points = np.array([xs, ys]).T
    conv_hull = ConvexHull(points)
    vertices = []
    if do_eeg_plot:
        plt.plot(points[:, 0], points[:, 1], 'o')
        for simplex in conv_hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

    for vertix in conv_hull.vertices:
        vertices.append(points[vertix])

    return np.array(vertices)


def ig_results_loc_scatter_maps(subject, freq_name, cfg, target_dfs, electrodes_pos, method, silent_chan):

    for target_state in [0, 1, 2]:
        # freq_name=FREQ
        # target_state=0
        imp = target_dfs[target_state][['electrode_name', f'perc_mean_mot_{target_state}']]
        electr_coord = pd.DataFrame(np.array([electrodes_pos[:, -1],
                                              electrodes_pos[:, 1].astype(float),
                                              electrodes_pos[:, 2].astype(float)]).T)

        electr_coord.columns = ['electrode_name', 'x', 'y']

        imp_coord = pd.merge(imp.assign(str_name=imp.electrode_name.astype(str)),
                             electr_coord.assign(str_name=electr_coord.electrode_name.astype(str)),
                             how='inner', on='str_name')[['str_name', 'x', 'y', f'perc_mean_mot_{target_state}']]
        imp_coord.columns = ['electrode_name', 'x', 'y', f'perc_mean_mot_{target_state}']

        xs = np.array(imp_coord.x).astype(float)
        ys = np.array(imp_coord.y).astype(float)
        zs = np.array(imp_coord[f'perc_mean_mot_{target_state}']).astype(float)
        xs_s = np.array(electr_coord.x).astype(float)[silent_chan]
        ys_s = np.array(electr_coord.y).astype(float)[silent_chan]

        fig, ax = plt.subplots()
        im = ax.scatter(xs, ys, s=30, c=zs*100)
        im = ax.scatter(xs_s, ys_s, s=30, c='red')
        fig.colorbar(im, ax=ax)

        ax.set_title(f'MLP: F. Mean Importance with {method} \n'
                     f'Subject {subject} - Target state {target_state} - Freq {freq_name}')
        fig.savefig(osp.join(cfg['outputs_path'], cfg['interpret_figures'], cfg['interpret_loc'],
                             f'{subject}_{method}_electrode_scatter_mean_motiv{target_state}_freq_{freq_name}.png'))
        plt.close('all')


def ig_results_loc_surf_maps(subject, freq_name, cfg, target_dfs, electrodes_pos, method, silent_chan, session_sufix=''):

    if session_sufix == '':
        electr_coord = pd.DataFrame(np.array([electrodes_pos[:, -1],
                                              electrodes_pos[:, 1].astype(float),
                                              electrodes_pos[:, 2].astype(float)]).T)
    else:
        names = np.array([i[3:] for i in electrodes_pos[:, 0]])
        electr_coord = pd.DataFrame(np.array([names.astype(str),
                                              electrodes_pos[:, 1].astype(float),
                                              electrodes_pos[:, 2].astype(float)]).T)
    electr_coord['is_silent'] = silent_chan
    electr_coord.columns = ['electrode_name', 'x', 'y', 'is_silent']

    # freq_name=FREQ
    # target_state=0
    for target_state in [0, 1, 2]:
        mean_imp = target_dfs[target_state][['electrode_name', f'perc_mean_mot_{target_state}']]
        median_imp = target_dfs[target_state][['electrode_name', f'perc_median_mot_{target_state}']]
        measure_names = ['mean', 'median']

        # i=0, imp=mean_imp
        for i, imp in enumerate([mean_imp, median_imp]):
            measure = measure_names[i]
            imp_coord = pd.merge(imp.assign(str_name=imp.electrode_name.astype(str)),
                                 electr_coord.assign(str_name=electr_coord.electrode_name.astype(str)),
                                 how='inner', on='str_name')[['str_name', 'is_silent', 'x', 'y',
                                                              f'perc_{measure}_mot_{target_state}']]
            imp_coord.columns = ['electrode_name', 'is_silent', 'x', 'y', f'perc_{measure}_mot_{target_state}']

            x = np.array(imp_coord.x).astype(float)
            xs = np.array(imp_coord[imp_coord.is_silent].x).astype(float)
            y = np.array(imp_coord.y).astype(float)
            ys = np.array(imp_coord[imp_coord.is_silent].y).astype(float)
            z = np.array(imp_coord[f'perc_{measure}_mot_{target_state}']).astype(float)

            # target grid to interpolate to
            if session_sufix == '':
                grid_x, grid_y = np.mgrid[-2:2:100j, -2:2:200j]
            else:
                grid_x, grid_y = np.mgrid[-8:8:100j, -9:10:200j]
            points = np.array([x, y]).T
            silent_points = np.array([xs, ys]).T
            values = z

            # interpolate
            zi = griddata(points, values, (grid_x, grid_y), method='cubic') if session_sufix == '' else \
                griddata(points, values, (grid_x, grid_y), method='linear')

            # plot
            plt.subplot(111)
            if session_sufix == '':
                plt.imshow(zi.T, extent=(-2.1, 2.1, -2.1, 2.1), origin='lower', cmap='viridis')
            else:
                plt.imshow(zi.T, extent=(-8.1, 8.1, -9.1, 10.1), origin='lower')
            plt.plot(points[:, 0], points[:, 1], 'x', ms=5, c='grey')
            plt.plot(silent_points[:, 0], silent_points[:, 1], 'o', ms=5, c='white')
            plt.colorbar()
            if session_sufix == '':
                plt.clim(0, 0.19)
            else:
                plt.clim(0, 0.09)
            med_str = '' if session_sufix == '' else f'{session_sufix} Medication '
            pd_sufix = '' if session_sufix == '' else '_pd'
            plt.title(f'MLP: F.{measure} Importance with {method} \nSubject {subject} {med_str}- '
                      f'Target state {target_state} - Freq {freq_name}')
            plt.savefig(osp.join(cfg['outputs_path'], cfg[f'interpret_figures{pd_sufix}'], cfg['interpret_loc'],
                                 f'{subject}{session_sufix}_{method}_electrode_surf_{measure}_motiv{target_state}'
                                 f'_freq_{freq_name}.png'))
            plt.close('all')


def input_power_surf(subject, freq_name, cfg, input_array, electrodes_pos, silent_chan, session_sufix=''):
    # input_array = test_input_tensor.detach().numpy()
    input_perc = input_array / input_array.sum(axis=1).reshape(-1, 1)
    pow_means = input_perc.mean(axis=0)
    pow_medians = np.median(input_perc, axis=0)

    if session_sufix == '':
        electr_coord = pd.DataFrame(np.array([electrodes_pos[:, -1],
                                              electrodes_pos[:, 1].astype(float),
                                              electrodes_pos[:, 2].astype(float)]).T)
    else:
        names = np.array([i[3:] for i in electrodes_pos[:, 0]])
        electr_coord = pd.DataFrame(np.array([names.astype(str),
                                              electrodes_pos[:, 1].astype(float),
                                              electrodes_pos[:, 2].astype(float)]).T)
    electr_coord['is_silent'] = silent_chan
    electr_coord.columns = ['electrode_name', 'x', 'y', 'is_silent']

    electr_coord['mean_pow'] = pow_means
    electr_coord['median_pow'] = pow_medians
    measure_names = ['mean', 'median']

    # measure = 'mean'
    for measure in measure_names:

        x = np.array(electr_coord.x).astype(float)
        xs = np.array(electr_coord[electr_coord.is_silent].x).astype(float)
        y = np.array(electr_coord.y).astype(float)
        ys = np.array(electr_coord[electr_coord.is_silent].y).astype(float)
        z = np.array(electr_coord[f'{measure}_pow']).astype(float)

        # target grid to interpolate to
        if session_sufix == '':
            grid_x, grid_y = np.mgrid[-2:2:100j, -2:2:200j]
        else:
            grid_x, grid_y = np.mgrid[-8:8:100j, -9:10:200j]
        points = np.array([x, y]).T
        silent_points = np.array([xs, ys]).T
        values = z

        # interpolate
        zi = griddata(points, values, (grid_x, grid_y), method='cubic')

        # plot
        plt.subplot(111)
        if session_sufix == '':
            plt.imshow(zi.T, extent=(-2.1, 2.1, -2.1, 2.1), origin='lower', cmap='plasma')
        else:
            plt.imshow(zi.T, extent=(-8.1, 8.1, -9.1, 10.1), origin='lower', cmap='plasma')
        plt.plot(points[:, 0], points[:, 1], 'x', ms=5, c='grey')
        plt.plot(silent_points[:, 0], silent_points[:, 1], 'o', ms=5, c='white')
        plt.colorbar()
        med_str = '' if session_sufix == '' else f'{session_sufix} Medication '
        pd_sufix = '' if session_sufix == '' else '_pd'
        plt.title(f'{measure} power - Subject {subject} {med_str}- Freq {freq_name}')
        plt.savefig(osp.join(cfg['outputs_path'], cfg[f'interpret_figures{pd_sufix}'], cfg['pow_loc'],
                             f'{subject}{session_sufix}_electrode_surf_{measure}_pow'
                             f'_freq_{freq_name}.png'))
        plt.close('all')
