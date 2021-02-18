import torch


class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, np_input, np_targets):
        self.np_input = torch.from_numpy(np_input)
        self.np_targets = torch.from_numpy(np_targets)

    def __len__(self):
        return self.np_input.shape[0]

    def __getitem__(self, idx):
        return self.np_input[idx], self.np_targets[idx]
