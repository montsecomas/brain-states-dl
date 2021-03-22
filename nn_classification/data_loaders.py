import torch
import numpy as np
from utils.utils import is_pd_patient, processed_data_path, processed_labels_path
from sklearn import preprocessing


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
