import numpy as np


def discard_channels(data):
    # discard silent channels
    # input data must correspond to one session (electrodes, ms, trials, categ)
    invalid_ch = np.logical_or(np.abs(data[:, :, 0, 0]).max(axis=1) == 0,
                               np.isnan(data[:, 0, 0, 0]))
    valid_ch = np.logical_not(invalid_ch)
    cleaned_data = data[valid_ch, :, :, :]
    N = valid_ch.sum()
    return cleaned_data, N


def freq_filter(ts, n_motiv, n_trials, T, N, n_bands=3):
    import scipy.signal as spsg
    freq_bands = ['alpha', 'beta', 'gamma']
    filtered_ts = np.zeros([n_bands, n_motiv, n_trials, T, N])
    for i_band in range(n_bands):
        # select band
        freq_band = freq_bands[i_band]

        # band-pass filtering (alpha, beta, gamma)
        n_order = 3
        sampling_freq = 500. # sampling rate

        if freq_band == 'alpha':
            low_f = 8./sampling_freq
            high_f = 15./sampling_freq
        elif freq_band == 'beta':
            # beta
            low_f = 15./sampling_freq
            high_f = 32./sampling_freq
        elif freq_band == 'gamma':
            # gamma
            low_f = 32./sampling_freq
            high_f = 80./sampling_freq
        else:
            raise NameError('unknown filter')

        # apply filter ts[n_motiv,n_trials,T,N]
        b, a = spsg.iirfilter(n_order, [low_f, high_f], btype='bandpass', ftype='butter')
        filtered_ts[:, i_band, :, :, :] = spsg.filtfilt(b, a, ts, axis=2)

    return filtered_ts


def get_ts(data, n_motiv, n_trials, ms, chan):
    # get time series for each block
    ts = np.zeros([n_motiv, n_trials, ms, chan])
    for i_motiv in range(n_motiv):
        for i_trial in range(n_trials):
            # swap axes for time and channels
            ts[i_motiv, i_trial, :, :] = data[:, :, i_trial, i_motiv].T

    return ts
