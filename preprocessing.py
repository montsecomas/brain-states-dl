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
        """
        Initizalize variables
        :param ts_data: dimensions (total_trials, t, red_n)
        :param pd_sub:
        """
        self.ts_data = ts_data
        self.pd_sub = pd_sub
        self.N_MOTIV = 3
        self.total_trials = ts_data.shape[0]
        self.ms = ts_data.shape[1]
        self.n_raw_features = ts_data.shape[2]
        self.sampling_freq = 500.
        self.n_bands = 3
        self.band_filter_order = 3
        self.bandpassed = self._filter_frequencies()
        self.ini_eeg_f = self._build_cor_cov_mat()
        self.mask_tri = np.tri(self.n_raw_features, self.n_raw_features, -1, dtype=np.bool_)

    def _filter_frequencies(self):
        """
        Decompose sequences in self.ts_data (total_trials, t, red_n) by different freq bands
        :return: decomposed series (n_bands, total_trials, t, red_n)
        """
        import scipy.signal as spsg
        freq_bands = ['alpha', 'beta', 'gamma']
        if len(freq_bands) != self.n_bands:
            raise ValueError('Rename frequency bands')
        freqs_ts = np.empty([0, self.total_trials, self.ms, self.n_raw_features])
        for i_band in range(self.n_bands):
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

    def _build_cor_cov_mat(self):
        EEG_FC = np.zeros([self.n_bands, self.total_trials, self.n_raw_features, self.n_raw_features])
        for i_band in range(self.n_bands):
            for i_trial in range(self.total_trials):
                ts_tmp = self.bandpassed[i_band, i_trial, :, :].copy()
                ts_tmp -= np.outer(np.ones(self.ms), ts_tmp.mean(0))
                EEG_FC[i_band, i_trial, :, :] = np.tensordot(ts_tmp, ts_tmp, axes=(0, 0)) / float(self.ms - 1)
        return EEG_FC

    def build_signal_dataset(self):
        """
        Power of signal within each sliding window (rectification by absolute value)
        :return: mean absolute values for each feature (n_bands, total_trials, n_raw_features)
        """
        return np.abs(self.bandpassed).mean(axis=-2)

    def build_cov_dataset(self):
        return self.ini_eeg_f[:, :, self.mask_tri].copy()

    def build_cor_dataset(self):
        eeg_f = self.ini_eeg_f.copy()
        for i_band in range(self.n_bands):
            for i_trial in range(self.total_trials):
                eeg_f[i_band, i_trial, :, :] /= np.sqrt(
                    np.outer(eeg_f[i_band, i_trial, :, :].diagonal(), eeg_f[i_band, i_trial, :, :].diagonal()))

        return eeg_f[:, :, self.mask_tri]


if __name__ == '__main__':
    for subject in [35]:
        sample = BrainStatesTrial(35)
        clean_data, clean_pks, is_pd = sample.run_pipeline()
        sample_featuring = BrainStatesFeaturing(clean_data, is_pd)
        signal_ds = sample_featuring.build_signal_dataset()
        cov_ds = sample_featuring.build_cov_dataset()
        cor_ds = sample_featuring.build_cor_dataset()
