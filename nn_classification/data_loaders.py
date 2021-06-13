import torch
import numpy as np
from utils.file_utils import is_pd_patient, processed_data_path, processed_labels_path
from sklearn import preprocessing
from torch.utils.data import DataLoader
import os.path as osp


def subject_nn_data(subject, healthy_subjects, pd_subjects, data_path, pd_dir, healthy_dir,
                    on_med=False, use_silent_channels=True, mask_value=None, feature_name=None, conv=False):
    is_pd = is_pd_patient(subject, healthy_subjects, pd_subjects)
    input_data = np.load(processed_data_path(subject_id=subject, is_pd=is_pd, use_silent_channels=use_silent_channels,
                                             feature_name=feature_name, data_path=data_path, pd_dir=pd_dir,
                                             healthy_dir=healthy_dir, conv=conv))
    np_labels = np.load(processed_labels_path(subject_id=subject, is_pd=is_pd, data_path=data_path,
                                              pd_dir=pd_dir, healthy_dir=healthy_dir))
    invalid_ch = np.load(processed_data_path(subject_id=subject, is_pd=is_pd, use_silent_channels=False,
                                             feature_name='silent_channels', data_path=data_path, pd_dir=pd_dir,
                                             healthy_dir=healthy_dir))

    if is_pd:
        on_med_ind = np.char.find(np_labels[:, 1], 'on') == 0
        ses_labels = np_labels[on_med_ind] if on_med else np_labels[~on_med_ind]
        le = preprocessing.LabelEncoder()
        le.fit(ses_labels[:, 1])
        targets = le.transform(ses_labels[:, 1])
        input_data = input_data[:, on_med_ind, :] if on_med else input_data[:, ~on_med_ind, :]
    else:
        le = preprocessing.LabelEncoder()
        le.fit(np_labels[:, 1])
        targets = le.transform(np_labels[:, 1])

    if conv:
        aux_invalid_ch = np.repeat(np.array([invalid_ch]), [3], axis=0).reshape(-1)
        ts_invalid_ch = np.tile(aux_invalid_ch[None], (input_data.shape[2], 1)).T
        invalid_mask = np.tile(ts_invalid_ch[None], (input_data.shape[0], 1, 1))
    else:
        invalid_mask = np.tile(invalid_ch[None, None], (input_data.shape[0], input_data.shape[1], 1))

    if use_silent_channels:
        if mask_value == '-1':
            # input_data[np.isnan(input_data)] = float(-1)
            input_data[invalid_mask] = float(-1)
        if mask_value == '0':
            # if conv:
            #     input_data[np.isnan(input_data)] = float(0)
            # else:
            input_data[invalid_mask] = float(0)
        elif mask_value == 'mean':
            inds = np.where(np.isnan(input_data))
            dim_means = np.nanmean(input_data, axis=2)
            input_data[inds] = np.take(dim_means, inds[1])

    return input_data, targets, np_labels


class FlatEEGDataset(torch.utils.data.Dataset):
    def __init__(self, np_input, np_targets):
        self.np_input = torch.from_numpy(np_input)
        self.np_targets = torch.from_numpy(np_targets).long()

    def __len__(self):
        return self.np_input.shape[0]

    def __getitem__(self, idx):
        return self.np_input[idx], self.np_targets[idx]


class FullEEGDataset(torch.utils.data.Dataset):
    def __init__(self, np_input, np_targets):
        self.np_input = torch.from_numpy(np_input)
        self.np_targets = torch.from_numpy(np_targets).long()

    def __len__(self):
        return self.np_input.shape[0]

    def __getitem__(self, idx):
        return self.np_input[idx], self.np_targets[idx]


class SingleSubjectNNData:
    def __init__(self, subject, classifier, cfg, read_silent_channels=True, force_read_split=False):
        self.subject = subject
        self.cfg = cfg
        self.force_read_split = force_read_split
        self.read_silent_channels = read_silent_channels
        if cfg['run_pd']:
            self.med_str = '-on-med' if cfg['model_on_med'] else '-off-med'
        else:
            self.med_str = ''
        self.split_idx_path = osp.join(cfg['outputs_path'],
                                       cfg['splits_path'],
                                       f'{subject}{self.med_str}-mlp.npy')

        self.freqs = ['alpha', 'beta', 'gamma']
        self.freqs_idx = [0, 1, 2]
        if classifier == 'mlp':
            self.input_dataset, self.targets, self.indices = self.mlp_full_ds()
            self.split_idx = int(self.input_dataset.shape[1] * 0.9)

    def mlp_full_ds(self):
        input_data, targets, long_labels = subject_nn_data(self.subject,
                                                           healthy_subjects=self.cfg['healthy_subjects'],
                                                           pd_subjects=self.cfg['pd_subjects'],
                                                           feature_name=self.cfg['pred_feature'],
                                                           data_path=self.cfg['data_path'],
                                                           pd_dir=self.cfg['pd_dir'],
                                                           healthy_dir=self.cfg['healthy_dir'],
                                                           on_med=self.cfg['model_on_med'],
                                                           use_silent_channels=self.read_silent_channels,
                                                           mask_value=self.cfg['mask_value'])

        if self.force_read_split or osp.exists(self.split_idx_path):
            indices = np.load(self.split_idx_path)
        else:
            indices = np.arange(input_data.shape[1])
            np.random.shuffle(indices)
            np.save(self.split_idx_path, indices)

        return input_data, targets, indices

    def mlp_ds_loaders(self, freq, random_test=False):
        train_data = FlatEEGDataset(np_input=self.input_dataset[freq, self.indices[:self.split_idx], :],
                                    np_targets=self.targets[self.indices[:self.split_idx]])
        val_data = FlatEEGDataset(np_input=self.input_dataset[freq, self.indices[self.split_idx:], :],
                                  np_targets=self.targets[self.indices[self.split_idx:]])

        # data loaders
        if random_test:
            train_loader = DataLoader(train_data, batch_size=len(train_data), shuffle=True, num_workers=0)
        else:
            train_loader = DataLoader(train_data, batch_size=self.cfg['batch_size'], shuffle=True, num_workers=0)
        val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=False, num_workers=0)

        return train_loader, val_loader
