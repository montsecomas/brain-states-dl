import os
import numpy as np
import scipy.io as sio
import multiprocessing as mp
from ini_montse.shared_process import freq_filter, discard_channels
import itertools

PD_N = 128
T = 1600
N_MOTIV = 3
TRIALS = 108
# TODO: parameterize, dimensions come from the input (?)


def define_subject_dir(i_sub):
    """
    Creates the directory if it doesn't exist
    :param i_sub: subject id
    :return: directory path
    """
    res_dir = "data/res_subject_" + str(i_sub) + "/"
    if not os.path.exists(res_dir):
        print("create directory:", res_dir)
        os.makedirs(res_dir)
    return res_dir


def read_data_ici(i_sub, subset='dataSorted'):
    """
    :param i_sub: subject id
    :param subset: key of the python dictionary corresponding to .mat file
    :return: numpy array corresponding to the specified key
    """
    subj_dir = define_subject_dir(i_sub)
    raw_readings = sio.loadmat("data/dataClean-ICA-" + str(i_sub) + "-T1.mat")
    # readings = raw_readings["dataSorted"][:, :, :, :3, :2]
    return raw_readings[subset], subj_dir


def process_data_sorted(data):
    """
    :param data: numpy array with dimensions (N, MS, TRIALS, STATE, MED)
        N: number of electrodes
        ms: Time sequence milliseconds
        TRIALS: number of trials per STATE
        STATE: number of different types of trials (6: 3 motivation states x 2 experiments stop-in/crossover)
        MED: 2: on & off medication
    :return: numpy array of dimension (med, rg, n_mot, MS, N)
        med: 1 if only off-med session, 2 otherwise
        rg: 2 - types of experiments: stop-in / crossover
        n_mot: 3 - number of motivation states
    """
    # source: signal - ms - trial - rg+moti - med
    on_med = data[:, :, :, :, 0]
    off_med = data[:, :, :, :, 1]

    _, N_on = discard_channels(on_med) # TODO: discard silent channels (silent electrodes by session)??
    if N_on == 0:
        subsets = [off_med]
    else:
        # REMARK: the order of the sessions is inverted (0- off med, 1- on med)
        subsets = [off_med, on_med]

    ts = np.empty(shape=(0, 2, N_MOTIV, TRIALS, T, PD_N))
    for ds in subsets:
        # on/off med - rg1/rg2 - motiv - trial - sec - signal
        ts_tmp_new = np.zeros((2, N_MOTIV, TRIALS, T, PD_N))

        # rg1:
        ts_tmp_new[0, :, :, :] = np.array(
            (ds[:, :, :, 0].T, ds[:, :, :, 1].T, ds[:, :, :, 2].T)
        )
        # rg2:
        ts_tmp_new[1, :, :, :] = np.array(
            (ds[:, :, :, 3].T, ds[:, :, :, 4].T, ds[:, :, :, 5].T)
        )

        ts = np.concatenate((ts, np.array([ts_tmp_new])))

    return ts


def define_labels(data, PD=True):
    """
    Label the ts array of a patient with an id and a label for the corresponding state
    :param PD: bool, True if data corresponds to PD patients, False otherwise
    :param data: time series in an array of dimensions (med, rg, n_motive, TRIALS, MS, N)
    :return: two arrays of dimensions (med x rg x n_motiv x trials, MS, N) and (med x rg x n_motiv x trials, 2)
    """
    if PD:
        if data.shape[0] > 1:
            id_list = [['off', 'on'], ['rg1', 'rg2'],
                       list(np.arange(N_MOTIV).astype(str)), list((np.arange(TRIALS) + 1).astype(str))]
        else:
            id_list = [['off'], ['rg1', 'rg2'],
                       list(np.arange(N_MOTIV).astype(str)), list((np.arange(TRIALS) + 1).astype(str))]
        n_electrodes = PD_N
    else:
        # NOT IMPLEMENTED CASE
        id_list = [['heal'], ['rg1', 'rg2'],
                   list(np.arange(N_MOTIV).astype(str)), list((np.arange(TRIALS) + 1).astype(str))]
        n_electrodes = 0

    ids = list(itertools.product(*id_list))
    str_ids = np.array(["-".join(comb) for comb in ids]).reshape(-1, 1)
    str_labels = np.array(["-".join([comb[0], comb[2]]) for comb in ids]).reshape(-1, 1)
    data_ids = np.concatenate((str_ids, str_labels), axis=1)

    flat_ts = data.reshape(-1, *data.shape[-2:])
    # TODO: hi ha nans

    return flat_ts, data_ids


def eeg_features(data, i_measure=0):
    """
    Compute features (EEG signal (0), covariance (1), or correlation (2))
    :param data: time series (trials, T, N)
    :param i_measure: value in {0, 1, 2} corresponding to the description
    :return:
    """
    # TODO: apply filter
    filtered_ts = freq_filter(ts=data, n_motiv=N_MOTIV, n_trials=TRIALS, T=T, N=PD_N)

    # TODO: compute features
    if i_measure == 0:  # power of signal within each sliding window (rectification by absolute value)
        # create the design matrix [samples,features]
        vect_features = np.abs(filtered_ts).mean(axis=2)

    else:  # covariance or correlation
        EEG_FC = np.zeros([N_MOTIV, TRIALS, PD_N,
                           PD_N])  # dynamic FC = covariance or Pearson correlation of signal within each sliding window
        for i_motiv in range(N_MOTIV):
            for i_trial in range(TRIALS):
                ts_tmp = filtered_ts[i_motiv, i_trial, :, :]
                ts_tmp -= np.outer(np.ones(T), ts_tmp.mean(0))
                EEG_FC[i_motiv, i_trial, :, :] = np.tensordot(ts_tmp, ts_tmp, axes=(0, 0)) / float(T - 1)
                if i_measure == 2:  # correlation, not covariance
                    EEG_FC[i_motiv, i_trial, :, :] /= np.sqrt(
                        np.outer(EEG_FC[i_motiv, i_trial, :, :].diagonal(), EEG_FC[i_motiv, i_trial, :, :].diagonal()))

        # vectorize the connectivity matrices to obtain the design matrix [samples,features]
        mask_tri = np.tri(PD_N, PD_N, -1, dtype=np.bool)  # mask to extract lower triangle of matrix
        vect_features = EEG_FC[:, :, mask_tri]

    return 0


def build_ts_dataset(subjects=[68], n_motiv=N_MOTIV):
    # subjects = [62, 65, 68]
    subjects = [68]
    # i_sub = 62
    # i_sub = 68
    # (68: 3.42 GB (on-off med), 62: 1.46GB (off med))
    # TODO: parallelize following loop
    for i_sub in subjects:
        # read data for current subject:
        raw_sorted, i_dir = read_data_ici(i_sub, subset='dataSorted')

        # keep on/off-med trials if exist (swap order), reshape to time series:
        processed = process_data_sorted(data=raw_sorted)

        # label time series (create id and label) and flatten dimensions:
        flat_sub_ts, sub_ids = define_labels(data=processed)

        # TODO: extract features (EEG's signal, covariance, correlation
        feat_ds = eeg_features(data=flat_sub_ts)

        # clean memory
        del raw_sorted, processed

    pass


def train_test_split():
    pass


def pd_model():
    pass

