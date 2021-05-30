import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import os.path as osp
import pandas as pd
from shapely.geometry import Point
import random
from scipy.interpolate import griddata


def importances_results_df(subject, freq_name, cfg, attrs, electrodes_pos, save_tables=False):
    mean_attrs = pd.DataFrame(np.array([np.mean(attr, axis=0) for attr in attrs]).T)
    median_attrs = pd.DataFrame(np.array([np.median(attr, axis=0) for attr in attrs]).T)
    mm_attrs = pd.concat([mean_attrs, median_attrs], axis=1)
    mean_attrs.columns = ['raw_mot_0', 'raw_mot_1', 'raw_mot_2']
    mm_attrs.columns = ['raw_mean_mot_0', 'raw_mean_mot_1', 'raw_mean_mot_2',
                        'raw_median_mot_0', 'raw_median_mot_1', 'raw_median_mot_2']
    mean_attrs['electrode_id'] = mm_attrs['electrode_id'] = (mean_attrs.index + 1).astype(int)

    electrodes_t_names = pd.DataFrame(np.array([electrodes_pos[:, 0], electrodes_pos[:, -1]]).T)
    electrodes_t_names.columns = ['electrode_id', 'electrode_name']
    electrodes_t_names['electrode_id'] = electrodes_t_names['electrode_id'].astype(int)

    mean_attrs_names = mean_attrs.merge(electrodes_t_names, on='electrode_id')
    mm_attrs_names = mm_attrs.merge(electrodes_t_names, on='electrode_id')

    target_dfs = []
    for target_state in [0, 1, 2]:
        mean_attrs_names[f'abs_mot_{target_state}'] = np.abs(mean_attrs_names[f'raw_mot_{target_state}'])
        mm_attrs_names[f'abs_mean_mot_{target_state}'] = np.abs(mm_attrs_names[f'raw_mean_mot_{target_state}'])
        mm_attrs_names[f'abs_median_mot_{target_state}'] = np.abs(mm_attrs_names[f'raw_median_mot_{target_state}'])

        mean_attrs_names[f'perc_mot_{target_state}'] = \
            mean_attrs_names[f'abs_mot_{target_state}'] / mean_attrs_names[f'abs_mot_{target_state}'].sum()
        mm_attrs_names[f'perc_mean_mot_{target_state}'] = \
            mm_attrs_names[f'abs_mean_mot_{target_state}'] / mm_attrs_names[f'abs_mean_mot_{target_state}'].sum()
        mm_attrs_names[f'perc_median_mot_{target_state}'] = \
            mm_attrs_names[f'abs_median_mot_{target_state}'] / mm_attrs_names[f'abs_median_mot_{target_state}'].sum()

        target_dfs.append(mm_attrs_names[['electrode_id', 'electrode_name',
                                          f'raw_mean_mot_{target_state}',
                                          f'raw_mean_mot_{target_state}',
                                          f'raw_mean_mot_{target_state}',
                                          f'raw_median_mot_{target_state}',
                                          f'raw_median_mot_{target_state}',
                                          f'raw_median_mot_{target_state}',
                                          f'perc_mean_mot_{target_state}',
                                          f'perc_median_mot_{target_state}']].
                          sort_values(f'perc_mean_mot_{target_state}', ascending=False))

        if save_tables:
            target_dfs[target_state].to_csv(osp.join(cfg['outputs_path'], cfg['interpret_imp_mlp'],
                                                     f'{subject}_mlp_importances_motiv{target_state}-freq_{freq_name}.csv'),
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
        # target_state=0
        imp = target_dfs[target_state][['electrode_name', f'perc_mean_mot_{target_state}']]
        electr_coord = pd.DataFrame(np.array([electrodes_pos[:, -1],
                                              electrodes_pos[:, 1].astype(float),
                                              electrodes_pos[:, 2].astype(float)]).T)
        electr_coord.columns = ['electrode_name', 'x', 'y']
        imp_coord = imp.merge(electr_coord, on='electrode_name')

        xs = np.array(electr_coord.x).astype(float)
        ys = np.array(electr_coord.y).astype(float)
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


def ig_results_loc_surf_maps(subject, freq_name, cfg, target_dfs, electrodes_pos, method, silent_chan):

    electr_coord = pd.DataFrame(np.array([electrodes_pos[:, -1],
                                          electrodes_pos[:, 1].astype(float),
                                          electrodes_pos[:, 2].astype(float)]).T)
    electr_coord.columns = ['electrode_name', 'x', 'y']

    # target_state=0
    for target_state in [0, 1, 2]:
        mean_imp = target_dfs[target_state][['electrode_name', f'perc_mean_mot_{target_state}']]
        median_imp = target_dfs[target_state][['electrode_name', f'perc_median_mot_{target_state}']]
        measure_names = ['mean', 'median']

        # i=0, imp=mean_imp
        for i, imp in enumerate([mean_imp, median_imp]):
            measure = measure_names[i]
            imp_coord = imp.merge(electr_coord, on='electrode_name')

            x = np.array(electr_coord.x).astype(float)
            y = np.array(electr_coord.y).astype(float)
            z = np.array(imp_coord[f'perc_{measure}_mot_{target_state}']).astype(float)

            # target grid to interpolate to
            grid_x, grid_y = np.mgrid[-2:2:100j, -2:2:200j]
            points = np.array([x, y]).T
            silent_points = points[silent_chan]
            values = z

            # interpolate
            zi = griddata(points, values, (grid_x, grid_y), method='cubic')

            # plot
            plt.subplot(111)
            plt.imshow(zi.T, extent=(-2.1, 2.1, -2.1, 2.1), origin='lower')
            plt.plot(points[:, 0], points[:, 1], 'x', ms=5, c='grey')
            plt.plot(silent_points[:, 0], silent_points[:, 1], 'o', ms=5, c='white')
            plt.title(f'MLP: F.{measure} Importance with {method} \nSubject {subject} - '
                      f'Target state {target_state} - Freq {freq_name}')
            plt.savefig(osp.join(cfg['outputs_path'], cfg['interpret_figures'], cfg['interpret_loc'],
                                 f'{subject}_{method}_electrode_surf_{measure}_motiv{target_state}_freq_{freq_name}.png'))
            plt.close('all')
