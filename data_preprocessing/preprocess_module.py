import sys
sys.path.append('/Users/mcomastu/TFM/brain-states-dl') # TODO: remove this first two lines


import os
import numpy as np
import scipy.io as sio
import itertools
from utils.file_utils import raw_input_path, subject_res_dir
from scipy.ndimage import gaussian_filter


class BrainStatesSubject:
    def __init__(self, i_sub, PD, cfg):
        """
        The class works under the assumption that the raw data's dimensions are sorted as follows:
        (N_features, T_milliseconds, trials, x), where x:
            can be a single dimension: block x session x motivation
            or split in two dimensions: motivation x block, session # TODO: define real order
        :param i_sub: subject id
        :param subset: key of the python dictionary corresponding to .mat file
        channels
        """
        self.i_sub = i_sub
        self.subset = cfg['mat_dict']
        self.use_silent_channels = cfg['use_silent_channels']
        self.N_MOTIV = 3
        self.PD = PD
        self.data_path = cfg['data_path']
        self.pd_dir = cfg['pd_dir']
        self.pd_ses_order = cfg['pd_ses_order']
        self.healthy_dir = cfg['healthy_dir']
        self.input_path = raw_input_path(i_sub, is_pd=self.PD, data_path=self.data_path, pd_dir=self.pd_dir,
                                         healthy_dir=self.healthy_dir)

        subject_string = 'Parkinson subject.' if self.PD else 'Healthy subject.'
        print('------------------------------------\nSubject', i_sub, '-', subject_string,
              '\n------------------------------------')

        if self.subset == 'dataSorted':
            self.raw_data = sio.loadmat(self.input_path)[self.subset]
        elif self.subset == 'ica':
            if self.PD:
                self.raw_data = sio.loadmat(self.input_path)['ic_data']
            else:
                self.raw_data = sio.loadmat(self.input_path)['ic_data3']

    def run_pipeline(self):
        self._define_subject_dir()
        clean_ts, invalid_ch = self._process_data_sorted()
        flat_sub_ts, sub_ids = self._define_labels(clean_ts)
        return flat_sub_ts, sub_ids, self.PD, invalid_ch

    def _define_subject_dir(self):
        """
        Creates the directory for the outputs if it doesn't exist
        :return: directory path
        """
        res_dir = subject_res_dir(subject_id=self.i_sub, is_pd=self.PD, data_path=self.data_path, pd_dir=self.pd_dir,
                                  healthy_dir=self.healthy_dir)

        if not os.path.exists(res_dir):
            print("create directory:", res_dir)
            os.makedirs(res_dir)

    def _discard_channels(self, data):
        # discard silent channels
        # input data must have flatten session dimension: (electrodes, ms, trials, motiv x session x block)
        if self.PD:
            invalid_ch_s0 = np.logical_or(np.abs(data[:, :, 0, 0, self.pd_ses_order[self.i_sub][0]]).max(axis=1) == 0,
                                          np.isnan(data[:, 0, 0, 0, self.pd_ses_order[self.i_sub][0]]))
            invalid_ch_s1 = np.logical_or(np.abs(data[:, :, 0, 0, self.pd_ses_order[self.i_sub][1]]).max(axis=1) == 0,
                                          np.isnan(data[:, 0, 0, 0, self.pd_ses_order[self.i_sub][1]]))

            invalid_ch = np.logical_or(invalid_ch_s0, invalid_ch_s1)
            valid_ch = np.logical_not(invalid_ch)
            cleaned_data = data[valid_ch, :, :, :, :]
            N = valid_ch.sum()
            print(N, 'healthy channels out of ', data.shape[0])

        else:
            invalid_ch_s0 = np.logical_or(np.abs(data[:, :, 0, 0]).max(axis=1) == 0,
                                          np.isnan(data[:, 0, 0, 0]))
            invalid_ch_s1 = np.logical_or(np.abs(data[:, :, 0, 1]).max(axis=1) == 0,
                                          np.isnan(data[:, 0, 0, 1]))

            invalid_ch = np.logical_or(invalid_ch_s0, invalid_ch_s1)
            valid_ch = np.logical_not(invalid_ch)
            cleaned_data = data[valid_ch, :, :, :]
            N = valid_ch.sum()
            print(N, 'healthy channels out of ', data.shape[0])

        return cleaned_data, N, invalid_ch

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
            if self.use_silent_channels:
                clean_channels, red_n = self.raw_data, n
                _, _, invalid_ch = self._discard_channels(self.raw_data)
            else:
                clean_channels, red_n, invalid_ch = self._discard_channels(self.raw_data)

            session_onmeds = clean_channels[:, :, :, :, self.pd_ses_order[self.i_sub][0]]
            session_offmeds = clean_channels[:, :, :, :, self.pd_ses_order[self.i_sub][1]]
            subsets = [session_onmeds, session_offmeds]
        else:
            if self.use_silent_channels:
                clean_channels, red_n = self.raw_data, n
                _, _, invalid_ch = self._discard_channels(self.raw_data)
            else:
                clean_channels, red_n, invalid_ch = self._discard_channels(self.raw_data)
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

        return ts, invalid_ch

    def _define_labels(self, data):
        """
        :param data: time series in an array of dimensions (sess, rg, n_motive, TRIALS, MS, N)
        :return: two arrays of dimensions (sess x rg x n_motiv x trials, MS, N) and (sess x rg x n_motiv x trials, 2)
        """
        if self.PD:
            id_list = [['on', 'off'], ['rg1', 'rg2'],
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
    def __init__(self, input_ts, input_labels, pd_sub, use_silent_channels=False):
        """
        Initizalize variables
        :param ts_data: dimensions (total_trials, t, red_n)
        :param pd_sub:
        """
        print('Applying band-pass filters and computing final features')
        self.use_silent_channels = use_silent_channels
        self.input_ts = input_ts
        self.input_labels = input_labels
        self.ms = self.input_ts.shape[1]
        self.n_raw_features = self.input_ts.shape[2]
        self.ts_data, self.ts_labels = self._clean_undone_experiments()
        self.pd_sub = pd_sub
        self.N_MOTIV = 3
        self.total_trials = self.ts_data.shape[0]
        self.sampling_freq = 500.
        self.n_bands = 3
        self.band_filter_order = 3
        self.bandpassed = self._filter_frequencies()
        if not self.use_silent_channels:
            self.ini_eeg_f = self._build_cor_cov_mat()
        self.mask_tri = np.tri(self.n_raw_features, self.n_raw_features, -1, dtype=np.bool_)

    def _clean_undone_experiments(self):
        invalid_exp = np.sum(np.sum(np.logical_or(np.isnan(self.input_ts), (self.input_ts == 0)), axis=1),
                             axis=1) == self.ms*self.n_raw_features
        valid_exp = np.logical_not(invalid_exp)
        print(np.sum(valid_exp), 'left observations out of', self.input_ts.shape[0])

        return self.input_ts[valid_exp, :, :], self.input_labels[valid_exp, :]

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
        """
        Use it only when clean_channels = True
        :return: covariance between mean absolute values for each feature (n_bands, total_trials, n_raw_features^2/2)
        """
        return self.ini_eeg_f[:, :, self.mask_tri].copy()

    def build_cor_dataset(self):
        """
        Use it only when clean_channels = True
        :return: correlation between mean absolute values for each feature (n_bands, total_trials, n_raw_features^2/2)
        """
        eeg_f = self.ini_eeg_f.copy()
        for i_band in range(self.n_bands):
            for i_trial in range(self.total_trials):
                eeg_f[i_band, i_trial, :, :] /= np.sqrt(
                    np.outer(eeg_f[i_band, i_trial, :, :].diagonal(), eeg_f[i_band, i_trial, :, :].diagonal()))

        return eeg_f[:, :, self.mask_tri]


def sequence_downsampling(data, ds_step_len, gauss_filter=True):
    if gauss_filter:
        return gaussian_filter(data, sigma=(0, 0, 1))[:, :, ::ds_step_len]
    else:
        return data[:, :, ::ds_step_len]
