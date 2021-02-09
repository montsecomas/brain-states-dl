import os
import numpy as np
import scipy.io as sio
import multiprocessing as mp
from utils_montse.shared_process import freq_filter, discard_channels
import itertools


class BrainStatesTrial:
    def __init__(self, i_sub, subset='dataSorted'):
        """
        The class works under the assumption that the raw data's dimensions are sorted as follows:
        (N_features, T_milliseconds, trials, x), where x:
            can be a single dimension: block x session x motivation
            or split in two dimensions: motivation x block, session # TODO: define real order
        :param i_sub: subject id
        :param subset: key of the python dictionary corresponding to .mat file
        """
        self.i_sub = i_sub
        self.subset = subset
        self.HS_LIST = np.arange(25, 36)
        self.PD_LIST = [58, 59, 62, 65, 68]
        self.N_MOTIV = 3
        if i_sub in self.HS_LIST:
            self.PD = False
        elif i_sub in self.PD_LIST:
            self.PD = True
        else:
            raise ValueError(('No data for subject '+str(self.i_sub)))

        if self.PD:
            self.input_path = "data/pd_sb/dataClean-ICA-" + str(self.i_sub) + "-T1.mat"
        else:
            self.input_path = "data/healthy_sb/dataClean-ICA3-" + str(self.i_sub) + "-T1.mat"

        self.raw_data = sio.loadmat(self.input_path)[self.subset]

    def run_pipeline(self):
        subj_dir = self._define_subject_dir()
        clean_ts = self._process_data_sorted()
        flat_sub_ts, sub_ids = self._define_labels(clean_ts)
        return flat_sub_ts, sub_ids, self.PD

    def _define_subject_dir(self):
        """
        Creates the directory for the outputs if it doesn't exist
        :return: directory path
        """
        if self.PD:
            res_dir = "data/pd_sb/res_subject_" + str(self.i_sub) + "/"
        else:
            res_dir = "data/healthy_sb/res_subject_" + str(self.i_sub) + "/"
        if not os.path.exists(res_dir):
            print("create directory:", res_dir)
            os.makedirs(res_dir)
        return res_dir

    def _discard_channels(self, data):
        # discard silent channels
        # input data must have flatten session dimension: (electrodes, ms, trials, motiv x session x block)
        invalid_ch = np.logical_or(np.abs(data[:, :, 0, 0]).max(axis=1) == 0,
                                   np.isnan(data[:, 0, 0, 0]))
        valid_ch = np.logical_not(invalid_ch)
        cleaned_data = data[valid_ch, :, :, :]
        N = valid_ch.sum()
        return cleaned_data, N

    def _process_data_sorted(self):
        """
        Filter silent channels and reorder dimensions of the dataset
        :return: numpy array of dimensions (sess, rg, n_mot, trials, MS, N)
            sess: 1 if only off-med/1 session, 2 otherwise
            rg: 2 - types of experiments: stop-in / crossover
            n_mot: 3 - number of motivation states
        """
        n, t, trials = self.raw_data.shape[0], self.raw_data.shape[1], self.raw_data.shape[2]
        if self.PD:
            # TODO 1: flatten session dimension (checking if both exist, paying attention to the order of dimensions)
            # TODO 2: filter silent channels
            red_n = n
            subsets = [0, 0]
        else:
            if self.subset == 'dataSorted':
                clean_channels, red_n = self._discard_channels(self.raw_data)
            else:
                clean_channels, red_n = self.raw_data, n
            session1 = clean_channels[:, :, :, 0::2]
            session2 = clean_channels[:, :, :, 1::2]
            subsets = [session1, session2]

        ts = np.empty(shape=(0, 2, self.N_MOTIV, trials, t, red_n))
        for ds in subsets:
            # session(1/2 or on/off) - rg1/rg2 - motiv - trial - sec - signal
            ts_tmp_new = np.zeros((2, self.N_MOTIV, trials, t, red_n))

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

    def _define_labels(self, data):
        """
        :param data: time series in an array of dimensions (sess, rg, n_motive, TRIALS, MS, N)
        :return: two arrays of dimensions (sess x rg x n_motiv x trials, MS, N) and (sess x rg x n_motiv x trials, 2)
        """
        if self.PD:
            if data.shape[0] > 1:
                id_list = [['off', 'on'], ['rg1', 'rg2'],
                           list(np.arange(self.N_MOTIV).astype(str)), list((np.arange(data.shape[3]) + 1).astype(str))]
            else:
                id_list = [['off'], ['rg1', 'rg2'],
                           list(np.arange(self.N_MOTIV).astype(str)), list((np.arange(data.shape[3]) + 1).astype(str))]
        else:
            id_list = [['heal1', 'heal2'], ['rg1', 'rg2'],
                       list(np.arange(self.N_MOTIV).astype(str)), list((np.arange(data.shape[3]) + 1).astype(str))]

        ids = list(itertools.product(*id_list))
        str_ids = np.array(["-".join(comb) for comb in ids]).reshape(-1, 1)
        if self.PD:
            str_labels = np.array(["-".join([comb[0], comb[2]]) for comb in ids]).reshape(-1, 1)
        else:
            str_labels = np.array(["-".join(['heal', comb[2]]) for comb in ids]).reshape(-1, 1)
        data_ids = np.concatenate((str_ids, str_labels), axis=1)

        flat_ts = data.reshape(-1, *data.shape[-2:])

        return flat_ts, data_ids


class BrainStatesFeaturing:
    def __init__(self, ts_data, pd_sub):
        self.ts_data = ts_data
        self.pd_sub = pd_sub
        self.N_MOTIV = 3
        self.total_trials = ts_data.shape[0]
        self.ms = ts_data.shape[1]
        self.n_raw_features = ts_data.shape[2]
        self.sampling_freq = 500.
        self.band_filter_order = 3

    def _filter_frequencies(self):
        import scipy.signal as spsg
        freq_bands = ['alpha', 'beta', 'gamma']
        freqs_ts = np.empty([0, self.total_trials, self.ms, self.n_raw_features])
        for i_band in range(len(freq_bands)):
            freq_band = freq_bands[i_band]

            if freq_band == 'alpha':
                low_f = 8./self.sampling_freq
                high_f = 15./self.sampling_freq
            elif freq_band == 'beta':
                # beta
                low_f = 15./self.sampling_freq
                high_f = 32./self.sampling_freq
            elif freq_band == 'gamma':
                # gamma
                low_f = 32./self.sampling_freq
                high_f = 80./self.sampling_freq
            else:
                raise NameError('unknown filter')

            b, a = spsg.iirfilter(self.band_filter_order, [low_f, high_f],
                                  btype='bandpass', ftype='butter', output='ba')
            # ts_data: (trials, t, n)
            filtered_ts = spsg.filtfilt(b, a, self.ts_data, axis=-2)
            freqs_ts = np.concatenate((freqs_ts, np.array([filtered_ts])))

        return freqs_ts

    # TODO: check following functions
    def build_datasets(self, eeg_seq_mean=True, eeg_mean_cov=True, eeg_mean_cor=True):
        filt = sample_featuring._filter_frequencies()
        datasets = np.empty([0, 0, 0, 0]) # TODO: define dimensions
        if eeg_seq_mean:
            ds = self._seq_eeg_mean(filt)
            datasets = np.concatenate((datasets, np.array([ds])))

        if eeg_mean_cov or eeg_mean_cor:
            eeg_f = self._build_cor_cov_mat(filt)
            mask_tri = np.tri(self.n_raw_features, self.n_raw_features, -1, dtype=np.bool)
            if eeg_mean_cov:
                ds = eeg_f[:, :, mask_tri]
                datasets = np.concatenate((datasets, np.array([ds])))
            if eeg_mean_cor:
                for i_trial in range(self.total_trials):
                    eeg_f[i_trial, :, :] /= np.sqrt(
                        np.outer(eeg_f[i_trial, :, :].diagonal(), eeg_f[i_trial, :, :].diagonal()))
                ds = eeg_f[:, :, mask_tri]
                datasets = np.concatenate((datasets, np.array([ds])))

        return datasets

    def _seq_eeg_mean(self, filtered_ts):
        return np.abs(filtered_ts).mean(axis=-2)

    def _build_cor_cov_mat(self, filtered_ts):
        EEG_FC = np.zeros([self.total_trials, self.n_raw_features, self.n_raw_features])
        for i_trial in range(self.total_trials):
            ts_tmp = filtered_ts[i_trial, :, :]
            ts_tmp -= np.outer(np.ones(self.ms), ts_tmp.mean(0))
            EEG_FC[i_trial, :, :] = np.tensordot(ts_tmp, ts_tmp, axes=(0, 0)) / float(self.ms - 1)
        return EEG_FC

    def _seq_eeg_cov(self, EEG_FC):
        mask_tri = np.tri(self.n_raw_features, self.n_raw_features, -1, dtype=np.bool)
        return EEG_FC[:, :, mask_tri]

    def _seq_eeg_corr(self, EEG_FC):
        for i_trial in range(self.total_trials):
            EEG_FC[i_trial, :, :] /= np.sqrt(
                np.outer(EEG_FC[i_trial, :, :].diagonal(), EEG_FC[i_trial, :, :].diagonal()))
        mask_tri = np.tri(self.n_raw_features, self.n_raw_features, -1, dtype=np.bool)
        return EEG_FC[:, :, mask_tri]


if __name__ == '__main__':
    for subject in [35]:
        sample = BrainStatesTrial(35)
        clean_data, clean_pks, is_pd = sample.run_pipeline()
        sample_featuring = BrainStatesFeaturing(clean_data, is_pd)
        # filt = sample_featuring._filter_frequencies()
        # filt.shape: (3, 1296, 1200, 47)
        # shapes (1296, 1200, 47), (1296, 2)
