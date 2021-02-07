import os
import numpy as np
import scipy.io as sio
import multiprocessing as mp
from ini_montse.shared_process import freq_filter, discard_channels

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


def process_data(data):
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

    _, N_on = discard_channels(on_med)
    if N_on == 0:
        subsets = [off_med]
    else:
        subsets = [on_med, off_med]

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

    # TODO: remove silent channels (silent electrodes)??

    return ts


def define_labels(data):
    return 0


def build_ts_dataset(subjects=[68], n_motiv=N_MOTIV):
    # subjects = [62, 65, 68]
    subjects = [68]
    # i_sub = 62
    # i_sub = 68
    # (68: 3.42 GB (on-off med), 62: 1.46GB (off med))
    for i_sub in subjects:
        # read data for current subject:
        raw_sorted, i_dir = read_data_ici(i_sub, subset='dataSorted')
        # keep on/off-med trials if exist, reshape to time series:
        # REMARK: second dimensions (type of experiment) should not be treated as a different class: TODO: 'mask'
        processed = process_data(data=raw_sorted)
        # label time series (create id and label)
        labeled_series = define_labels(data=processed)


    pass


def train_test_split():
    pass


def pd_model():
    pass

