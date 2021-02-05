#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 17:34:37 2018
"""

import os
import numpy as np
import scipy.io as sio
import multiprocessing as mp
import process


#%% general info

if __name__ == "__main__":
    subjects = [55]
    print("Parallelizing with", 6, "subprocesses")
    pool = mp.Pool(6)

    for i_sub in subjects:

        res_dir = "lluis_subNS" + str(i_sub) + "/"
        if not os.path.exists(res_dir):
            print("create directory:", res_dir)
            os.makedirs(res_dir)

        #
        cmapcolours = ["Blues", "Greens", "Oranges"]
        listcolours = ["b", "g", "r"]

        # measure_labels = ['pow', 'cov', 'corr']
        measure_labels = ["pow"]
        n_measures = len(measure_labels)

        freq_bands = ["alpha", "beta", "gamma"]
        n_bands = len(freq_bands)

        #%% load data

        #    ts = sio.loadmat('./cleanData18092019/dataClean-25-T1.mat')['dataSorted']
        n_motiv = 2  # N of motivation levels is actually 3, but we add the 3 motivation levels from session 2 to this
        print("loading")
        readings = sio.loadmat("./dataClean-ICA-" + str(i_sub) + "-T1.mat")[
            "dataSorted"
        ][
            :, :, :, :3, :2
        ]  # [N,T,n_trials,motiv,session]
        #    ts_tmp = sio.loadmat('../dataBackupMTLnew/cleanData/dataClean-ICA-'+str(i_sub)+'-T1.mat')['ic_data'][:,:,:,:3,1] # [N,T,n_trials,motiv]
        # ts_tmp = sio.loadmat('../cleanData18092019/dataClean-'+str(i_sub)+'-T1.mat')['dataSorted'][:,:,:,:3,0] # [N,T,n_trials,motiv]

        # Collapse 5th dimension into 4th dimension to compare across sessions
        print(readings.shape)
        ts_tmp_new = np.zeros((384, 1600, 108, n_motiv))
        ts_tmp = np.zeros((384, 1600, 108, n_motiv))

        ts_tmp_new[:, :, :, 0] = np.concatenate(
            (readings[:, :, :, 0, 0], readings[:, :, :, 1, 0], readings[:, :, :, 2, 0])
        )
        ts_tmp_new[:, :, :, 1] = np.concatenate(
            (readings[:, :, :, 0, 1], readings[:, :, :, 1, 1], readings[:, :, :, 2, 1])
        )

        ts_tmp = ts_tmp_new

        N = ts_tmp.shape[0]  # number of channels
        n_trials = 108  # number of trials per block
        T = 1600  # trial duration

        # discard silent channels
        invalid_ch = np.logical_or(
            np.abs(ts_tmp[:, :, 0, 0]).max(axis=1) == 0, np.isnan(ts_tmp[:, 0, 0, 0])
        )
        valid_ch = np.logical_not(invalid_ch)
        ts_tmp = ts_tmp[valid_ch, :, :, :]
        N = valid_ch.sum()

        # get time series for each block
        ts = np.zeros([n_motiv, n_trials, T, N])
        for i_motiv in range(n_motiv):
            for i_trial in range(n_trials):
                # swap axes for time and channels
                ts[i_motiv, i_trial, :, :] = ts_tmp[:, :, i_trial, i_motiv].T

        del ts_tmp  # clean memory

        mask_tri = np.tri(
            N, N, -1, dtype=np.bool
        )  # mask to extract lower triangle of matrix

        #%% get channel positions for plot

        # node positions for circular layout with origin at bottom
        var_dict = np.genfromtxt("GSN129.sfp")

        x_sensor = var_dict[:, 1]
        y_sensor = var_dict[:, 2]

        print(x_sensor)
        print(y_sensor)

        # positions of sensors
        pos_circ = dict()
        for i in range(int(N / 3)):
            pos_circ[i] = np.array([x_sensor[i], y_sensor[i]])

        # channel labels
        ch_labels = dict()
        for i in range(int(N / 3)):
            ch_labels[i] = i + 1

        # matrices to retrieve input/output channels from connections in support network
        row_ind = np.repeat(np.arange(N).reshape([N, -1]), N, axis=1)
        col_ind = np.repeat(np.arange(N).reshape([-1, N]), N, axis=0)
        row_ind = row_ind[mask_tri]
        col_ind = col_ind[mask_tri]

        #%% run process for each combination of frequency and method
        print("Parallelizing")
        results = pool.map(
            process.f,
            [
                # ("alpha", "pow", ts, N, T, n_motiv, n_trials, res_dir),
                # ("beta", "pow", ts, N, T, n_motiv, n_trials, res_dir),
                # ("gamma", "pow", ts, N, T, n_motiv, n_trials, res_dir),
                ("alpha", "cov", ts, N, T, n_motiv, n_trials, res_dir),
                ("beta", "cov", ts, N, T, n_motiv, n_trials, res_dir),
                ("gamma", "cov", ts, N, T, n_motiv, n_trials, res_dir),
                ("alpha", "corr", ts, N, T, n_motiv, n_trials, res_dir),
                ("beta", "corr", ts, N, T, n_motiv, n_trials, res_dir),
                ("gamma", "corr", ts, N, T, n_motiv, n_trials, res_dir),
            ],
        )

        #print("Running process")
        #process.f(("alpha", "cov", ts, N, T, n_motiv, n_trials, res_dir))