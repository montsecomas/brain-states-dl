import torch
import numpy as np
from utils.utils import is_pd_patient, processed_data_path, processed_labels_path
from sklearn import preprocessing


def subject_nn_data(subject, healthy_subjects, pd_subjects, feature_name, data_path, pd_dir, healthy_dir,
                    use_silent_channels=True):
    is_pd = is_pd_patient(subject, healthy_subjects, pd_subjects)
    input_data = np.load(processed_data_path(subject_id=subject, is_pd=is_pd, feature_name=feature_name,
                                             data_path=data_path, pd_dir=pd_dir, healthy_dir=healthy_dir))
    np_labels = np.load(processed_labels_path(subject_id=subject, is_pd=is_pd, data_path=data_path,
                                              pd_dir=pd_dir, healthy_dir=healthy_dir))
    le = preprocessing.LabelEncoder()
    le.fit(np_labels[:, 1])
    # le.classes_
    targets = le.transform(np_labels[:, 1])

    return input_data, targets


class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, np_input, np_targets):
        self.np_input = torch.from_numpy(np_input)
        self.np_targets = torch.from_numpy(np_targets).long()

    def __len__(self):
        return self.np_input.shape[0]

    def __getitem__(self, idx):
        return self.np_input[idx], self.np_targets[idx]
