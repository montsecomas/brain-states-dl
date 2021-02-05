# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 19:27:54 2020
@author: lluis
"""

import scipy.signal as spsg
import sklearn.metrics as skm
import matplotlib.pyplot as pp
import numpy as np
import sklearn.model_selection as skms
import sklearn.linear_model as skllm
import sklearn.neighbors as sklnn
import sklearn.preprocessing as skprp
import sklearn.pipeline as skppl
import sklearn.feature_selection as skfs
import scipy.stats as stt
import networkx as nx

#%% routine to be parallelized
def f(c):
    measure_labels = ["pow", "cov", "corr"]
    freq_bands = ["alpha", "beta", "gamma"]
    cv_schem = skms.StratifiedShuffleSplit(n_splits=1, test_size=0.2)

    print(c)
    freq_band, measure, ts, N, T, n_motiv, n_trials, res_dir = c
    i_measure = measure_labels.index(measure)
    i_band = freq_bands.index(freq_band)

    sub_id = (freq_band, measure)

    cmapcolours = ["Blues", "Greens", "Oranges"]

    mask_tri = np.tri(
        N, N, -1, dtype=np.bool
    )  # mask to extract lower triangle of matrix

    # matrices to retrieve input/output channels from connections in support network
    row_ind = np.repeat(np.arange(N).reshape([N, -1]), N, axis=1)
    col_ind = np.repeat(np.arange(N).reshape([-1, N]), N, axis=0)
    row_ind = row_ind[mask_tri]
    col_ind = col_ind[mask_tri]

    # MLR adapted for recursive feature elimination (RFE)
    class RFE_pipeline(skppl.Pipeline):
        def fit(self, X, y=None, **fit_params):
            """simply extends the pipeline to recover the coefficients (used by RFE) from the last element (the classifier)
            """
            super(RFE_pipeline, self).fit(X, y, **fit_params)
            self.coef_ = self.steps[-1][-1].coef_
            return self

    c_MLR = RFE_pipeline(
        [
            ("std_scal", skprp.StandardScaler()),
            (
                "clf",
                skllm.LogisticRegression(
                    C=10,
                    penalty="l2",
                    multi_class="multinomial",
                    solver="lbfgs",
                    max_iter=500,
                ),
            ),
        ]
    )

    # nearest neighbor
    c_1NN = sklnn.KNeighborsClassifier(
        n_neighbors=1, algorithm="brute", metric="correlation"
    )

    # cross-validation scheme
    cv_schem = skms.StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    n_rep = 4  # number of repetitions

    # RFE wrappers
    RFE_pow = skfs.RFE(c_MLR, n_features_to_select=3, verbose=True)
    RFE_FC = skfs.RFE(c_MLR, n_features_to_select=90, verbose=True)

    n_bands = len(freq_bands)
    n_measures = len(measure_labels)

    # record classification performance
    perf = np.zeros([n_bands, n_measures, n_rep, 2])  # (last index: MLR/1NN)
    perf_shuf = np.zeros([n_bands, n_measures, n_rep, 2])  # (last index: MLR/1NN)
    conf_matrix = np.zeros(
        [n_bands, n_measures, n_rep, 2, n_motiv, n_motiv]
    )  # (fourthindex: MLR/1NN)
    rk_pow = np.zeros(
        [n_bands, n_rep, N], dtype=np.int
    )  # RFE rankings for power (N feature)
    rk_FC = np.zeros(
        [n_bands, 2, n_rep, int(N * (N - 1) / 2)], dtype=np.int
    )  # RFE rankings for FC-type measures (N(N-1)/2 feature)
    pearson_corr_rk = np.zeros(
        [n_bands, n_measures, int(n_rep * (n_rep - 1) / 2)]
    )  # stability of rankings measured by Pearson correlation

    # band-pass filtering (alpha, beta, gamma)
    n_order = 3
    sampling_freq = 500.0  # sampling rate

    if freq_band == "alpha":
        low_f = 8.0 / sampling_freq
        high_f = 12.0 / sampling_freq
    elif freq_band == "beta":
        # beta
        low_f = 15.0 / sampling_freq
        high_f = 30.0 / sampling_freq
    elif freq_band == "gamma":
        # gamma
        low_f = 40.0 / sampling_freq
        high_f = 80.0 / sampling_freq
    else:
        raise NameError("unknown filter")

    print(sub_id, "low_f: ", low_f)
    print(sub_id, "high_f:", high_f)

    # apply filter
    b, a = spsg.iirfilter(n_order, [low_f, high_f], btype="bandpass", ftype="butter")

    print(sub_id, b, a)
    filtered_ts = spsg.filtfilt(b, a, ts, axis=2)

    print(sub_id, "frequency band, measure:", freq_band, measure_labels[i_measure])

    # need safe margins to remove filtering effect? seems like no
    if False:
        pp.plot(filtered_ts[0, 0, :, 0])

    if (
        i_measure == 0
    ):  # power of signal within each sliding window (rectification by absolute value)
        # create the design matrix [samples,features]
        vect_features = np.abs(filtered_ts).mean(axis=2)

    else:  # covariance or correlation
        EEG_FC = np.zeros(
            [n_motiv, n_trials, N, N]
        )  # dynamic FC = covariance or Pearson correlation of signal within each sliding window
        for i_motiv in range(n_motiv):
            for i_trial in range(n_trials):
                ts_tmp = filtered_ts[i_motiv, i_trial, :, :]
                ts_tmp -= np.outer(np.ones(T), ts_tmp.mean(0))
                EEG_FC[i_motiv, i_trial, :, :] = np.tensordot(
                    ts_tmp, ts_tmp, axes=(0, 0)
                ) / float(T - 1)
                if i_measure == 2:  # correlation, not covariance
                    EEG_FC[i_motiv, i_trial, :, :] /= np.sqrt(
                        np.outer(
                            EEG_FC[i_motiv, i_trial, :, :].diagonal(),
                            EEG_FC[i_motiv, i_trial, :, :].diagonal(),
                        )
                    )

        # vectorize the connectivity matrices to obtain the design matrix [samples,features]
        vect_features = EEG_FC[:, :, mask_tri]

    # labels of sessions for classification (train+test)
    labels = np.zeros(
        [n_motiv, n_trials], dtype=np.int
    )  # 0 = M0, 1 = M1, 2 = M2; 1 = M0 (S2), 2 = M1 (S2), 3 = M2 (S3)
    labels[1, :] = 1
    # labels[2,:] = 2;

    # vectorize dimensions motivation levels and trials
    mask_motiv_trials = np.ones([n_motiv, n_trials], dtype=np.bool)
    vect_features = vect_features[mask_motiv_trials, :]
    labels = labels[mask_motiv_trials]

    # replace all NaNs by the mean in the column
    vect_features = np.where(
        np.isnan(vect_features),
        np.ma.array(vect_features, mask=np.isnan(vect_features)).mean(axis=0),
        vect_features,
    )

    print(sub_id, "mask_motiv_trials:", mask_motiv_trials.shape)
    print(sub_id, "labels:", labels)

    ################
    # repeat classification for several splits for indices of sliding windows (train/test sets)
    for i_rep in range(n_rep):
        for ind_train, ind_test in cv_schem.split(
            vect_features, labels
        ):  # false loop, just 1

            print(sub_id, "fitting c_MLR")

            # train and test for original data
            c_MLR.fit(vect_features[ind_train, :], labels[ind_train])
            perf[i_band, i_measure, i_rep, 0] = c_MLR.score(
                vect_features[ind_test, :], labels[ind_test]
            )

            print(sub_id, "done")

            conf_matrix[i_band, i_measure, i_rep, 0, :, :] += skm.confusion_matrix(
                y_true=labels[ind_test],
                y_pred=c_MLR.predict(vect_features[ind_test, :]),
            )

            print(sub_id, "fitting c_1NN")

            c_1NN.fit(vect_features[ind_train, :], labels[ind_train])
            perf[i_band, i_measure, i_rep, 1] = c_1NN.score(
                vect_features[ind_test, :], labels[ind_test]
            )
            conf_matrix[i_band, i_measure, i_rep, 1, :, :] += skm.confusion_matrix(
                y_true=labels[ind_test],
                y_pred=c_1NN.predict(vect_features[ind_test, :]),
            )

            print(sub_id, "done")

            # shuffled performance distributions
            shuf_labels = np.random.permutation(labels)

            print(sub_id, "fitting shuffled c_MLR")

            c_MLR.fit(vect_features[ind_train, :], shuf_labels[ind_train])
            perf_shuf[i_band, i_measure, i_rep, 0] = c_MLR.score(
                vect_features[ind_test, :], shuf_labels[ind_test]
            )

            print(sub_id, "done")
            print(sub_id, "fitting shuffled c_1NN")

            c_1NN.fit(vect_features[ind_train, :], shuf_labels[ind_train])
            perf_shuf[i_band, i_measure, i_rep, 1] = c_1NN.score(
                vect_features[ind_test, :], shuf_labels[ind_test]
            )
            print(sub_id, "done")

            # RFE for MLR
            if i_measure == 0:  # power
                RFE_pow.fit(vect_features[ind_train, :], labels[ind_train])
                rk_pow[i_band, i_rep, :] = RFE_pow.ranking_
            else:  # covariance or correlation
                print(sub_id, "fitting RFE_FC")
                RFE_FC.fit(vect_features[ind_train, :], labels[ind_train])
                rk_FC[i_band, i_measure - 1, i_rep, :] = RFE_FC.ranking_
                print(sub_id, "done")

    # check stability RFE rankings
    for i_band in range(n_bands):
        for i_measure in range(n_measures):
            i_cnt = 0
            for i_rep1 in range(n_rep):
                for i_rep2 in range(i_rep1):
                    pearson_corr_rk[i_band, 0, i_cnt] = stt.pearsonr(
                        rk_pow[i_band, i_rep1, :], rk_pow[i_band, i_rep2, :]
                    )[0]
                    pearson_corr_rk[i_band, 1, i_cnt] = stt.pearsonr(
                        rk_FC[i_band, 0, i_rep1, :], rk_FC[i_band, 0, i_rep2, :]
                    )[0]
                    pearson_corr_rk[i_band, 2, i_cnt] = stt.pearsonr(
                        rk_FC[i_band, 1, i_rep1, :], rk_FC[i_band, 1, i_rep2, :]
                    )[0]
                    i_cnt += 1

    # save results
    np.save(res_dir + "perf.npy", perf)
    np.save(res_dir + "perf_shuf.npy", perf_shuf)
    np.save(res_dir + "conf_matrix.npy", conf_matrix)
    np.save(res_dir + "rk_pow.npy", rk_pow)
    np.save(res_dir + "rk_FC.npy", rk_FC)
    np.save(res_dir + "pearson_corr_rk.npy", pearson_corr_rk)

    #%% plots
    fmt_grph = "png"

    labels = np.zeros(
        [n_motiv, n_trials], dtype=np.int
    )  # 0 = M0, 1 = M1, 2 = M2; 1 = M0 (S2), 2 = M1 (S2), 3 = M2 (S3)
    labels[1, :] = 1

    measure_label = measure_labels[i_measure]

    # the chance level is defined as the trivial classifier that predicts the label with more occurrences
    chance_level = np.max(np.unique(labels, return_counts=True)[1]) / labels.size

    # plot performance and surrogate
    pp.figure(figsize=[4, 3])
    pp.axes([0.2, 0.2, 0.7, 0.7])
    pp.violinplot(perf[i_band, i_measure, :, 0], positions=[-0.2], widths=[0.3])
    pp.violinplot(perf[i_band, i_measure, :, 1], positions=[0.2], widths=[0.3])
    pp.violinplot(perf_shuf[i_band, i_measure, :, 0], positions=[0.8], widths=[0.3])
    pp.violinplot(perf_shuf[i_band, i_measure, :, 1], positions=[1.2], widths=[0.3])
    pp.plot([-1, 2], [chance_level] * 2, "--k")
    pp.axis(xmin=-0.6, xmax=1.6, ymin=0, ymax=1.05)
    pp.xticks([0, 1], ["Pearson Correlation", "surrogate"], fontsize=8)
    pp.ylabel("accuracy_" + freq_band + "_" + measure_label, fontsize=8)
    pp.title(freq_band + ", " + measure_label)
    print(
        sub_id,
        "Saving file",
        res_dir + "accuracy_" + freq_band + "_" + measure_label + "." + fmt_grph,
    )
    pp.savefig(
        res_dir + "accuracy_" + freq_band + "_" + measure_label + "." + fmt_grph,
        format=fmt_grph,
    )
    pp.close()

    # plot confusion matrix for MLR
    pp.figure(figsize=[4, 3])
    pp.axes([0.2, 0.2, 0.7, 0.7])
    pp.imshow(
        conf_matrix[i_band, i_measure, :, 0, :, :].mean(0),
        vmin=0,
        cmap=cmapcolours[i_band],
    )
    pp.colorbar()
    pp.xlabel("true label", fontsize=8)
    pp.ylabel("predicted label", fontsize=8)
    pp.title(freq_band + ", " + measure_label)
    pp.savefig(
        res_dir + "conf_mat_MLR_" + freq_band + "_" + measure_label + "." + fmt_grph,
        format=fmt_grph,
    )
    pp.close()

    # plot RFE support network
    pp.figure(figsize=[10, 10])
    pp.axes([0.05, 0.05, 0.95, 0.95])
    pp.axis("off")
    if i_measure == 0:  # power
        list_best_feat = np.argsort(rk_pow[i_band, :, :].mean(0))[
            :10
        ]  # select 10 best features
        node_color_aff = []
        g = nx.Graph()
        for i in range(N):
            g.add_node(i)
            if i in list_best_feat:
                node_color_aff += ["red"]
            else:
                node_color_aff += ["orange"]
        # nx.draw_networkx_nodes(g,pos=pos_circ,node_color=node_color_aff)
        # nx.draw_networkx_labels(g,pos=pos_circ,labels=ch_labels)
    else:  # covariance or correlation
        list_best_feat = np.argsort(rk_FC[i_band, i_measure - 1, :, :].mean(0))[
            :20
        ]  # select 20 best features
        g = nx.Graph()
        for i in range(N):
            g.add_node(i)
        node_color_aff = ["orange"] * N
        list_ROI_from_to = (
            []
        )  # list of input/output ROIs involved in connections of support network
        for ij in list_best_feat:
            g.add_edge(col_ind[ij], row_ind[ij])
        # nx.draw_networkx_nodes(g,pos=pos_circ,node_color=node_color_aff)
        # nx.draw_networkx_labels(g,pos=pos_circ,labels=ch_labels)
        # nx.draw_networkx_edges(g,pos=pos_circ,edges=g.edges(),edge_color=listcolours[i_band])
    pp.title(freq_band)
    pp.savefig(
        res_dir + "support_net_RFE_" + freq_band + "_" + measure_label + "." + fmt_grph,
        format=fmt_grph,
    )
    pp.close()

    # plot stability of RFE rankings
    pp.figure(figsize=[4, 3])
    pp.axes([0.2, 0.2, 0.7, 0.7])
    pp.violinplot(
        pearson_corr_rk[i_band, :, :].T, positions=range(n_measures), widths=[0.4] * 3
    )
    pp.axis(ymin=0, ymax=1)
    pp.xticks(range(n_measures), measure_labels, fontsize=8)
    pp.ylabel("Pearson between rankings", fontsize=8)
    pp.title(freq_band)
    pp.savefig(
        res_dir + "stab_RFE_rankings_" + freq_band + "." + fmt_grph, format=fmt_grph
    )
    pp.close()