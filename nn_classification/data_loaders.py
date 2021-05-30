import torch
import numpy as np
from utils.file_utils import is_pd_patient, processed_data_path, processed_labels_path
from sklearn import preprocessing
from torch.utils.data import DataLoader
import os.path as osp


def subject_nn_data(subject, healthy_subjects, pd_subjects, data_path, pd_dir, healthy_dir,
                    use_silent_channels=True, mask_value=None, feature_name=None, conv=False):
    is_pd = is_pd_patient(subject, healthy_subjects, pd_subjects)
    input_data = np.load(processed_data_path(subject_id=subject, is_pd=is_pd, use_silent_channels=use_silent_channels,
                                             feature_name=feature_name, data_path=data_path, pd_dir=pd_dir,
                                             healthy_dir=healthy_dir, conv=conv))
    np_labels = np.load(processed_labels_path(subject_id=subject, is_pd=is_pd, data_path=data_path,
                                              pd_dir=pd_dir, healthy_dir=healthy_dir))
    le = preprocessing.LabelEncoder()
    le.fit(np_labels[:, 1])
    targets = le.transform(np_labels[:, 1])

    if use_silent_channels:
        if mask_value == '-1':
            input_data[np.isnan(input_data)] = float(-1)
        if mask_value == '0':
            input_data[np.isnan(input_data)] = float(0)
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
        self.split_idx_path = osp.join(self.cfg['outputs_path'],
                                       self.cfg['splits_path'],
                                       f'{self.subject}-mlp.npy')

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
                                                           use_silent_channels=self.read_silent_channels,
                                                           mask_value=self.cfg['mask_value'])

        if self.force_read_split or osp.exists(self.split_idx_path):
            indices = np.load(self.split_idx_path)
        else:
            indices = np.arange(input_data.shape[1])
            np.random.shuffle(indices)
            np.save(self.split_idx_path, indices)

        return input_data, targets, indices

    def mlp_ds_loaders(self, freq):
        train_data = FlatEEGDataset(np_input=self.input_dataset[freq, self.indices[:self.split_idx], :],
                                    np_targets=self.targets[self.indices[:self.split_idx]])
        val_data = FlatEEGDataset(np_input=self.input_dataset[freq, self.indices[self.split_idx:], :],
                                  np_targets=self.targets[self.indices[self.split_idx:]])

        # data loaders
        train_loader = DataLoader(train_data, batch_size=self.cfg['batch_size'], shuffle=True, num_workers=0)
        val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=False, num_workers=0)

        return train_loader, val_loader
