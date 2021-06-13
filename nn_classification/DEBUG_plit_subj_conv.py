import sys
sys.path.append('/Users/mcomastu/TFM/brain-states-dl') # TODO: remove this first two lines


from torch.utils.data import DataLoader
from nn_classification.data_loaders import FullEEGDataset, subject_nn_data
from nn_classification.pl_module import LitConvClassifier
from utils.file_utils import load_cfg
import numpy as np
from scipy.ndimage import gaussian_filter
from torch import nn
import os.path as osp


if __name__ == '__main__':
    cfg = load_cfg()
    run_pd = cfg['run_pd']
    subjects_list = cfg['pd_subjects'] if run_pd else cfg['healthy_subjects']
    if run_pd:
        med_str = '-on-med' if cfg['model_on_med'] else '-off-med'
    else:
        med_str = ''

    subject = 25
    input_data, targets, long_labels = subject_nn_data(subject,
                                                       healthy_subjects=cfg['healthy_subjects'],
                                                       pd_subjects=cfg['pd_subjects'],
                                                       feature_name=cfg['pred_feature'],
                                                       data_path=cfg['data_path'],
                                                       pd_dir=cfg['pd_dir'],
                                                       healthy_dir=cfg['healthy_dir'],
                                                       use_silent_channels=True,
                                                       mask_value='0',
                                                       conv=True)

    # train-val split
    split_idx_path = osp.join(cfg['outputs_path'], cfg['splits_path'], f'{subject}{med_str}-mlp.npy')

    if osp.exists(split_idx_path):
        indices = np.load(split_idx_path)
    else:
        indices = np.arange(input_data.shape[1])
        np.random.shuffle(indices)
        np.save(split_idx_path, indices)

    split_idx = int(input_data.shape[1] * 0.9)

    if cfg['gamma_freq']:
        input_data = input_data[:, 120:, :]
        freq_prefix = 'freq-gamma-'
    else:
        freq_prefix = ''

    # ts downsampling
    # import matplotlib.pyplot as plt
    # plt.plot(np.arange(1200), input_data[0,0,:])
    # plt.plot(np.arange(1200)[::5], input_data[0,0,::5])
    # plt.plot(np.arange(1200), gaussian_filter(input_data, sigma=(0, 0, 1))[0,0,:])
    # plt.plot(np.arange(1200)[::5], gaussian_filter(input_data, sigma=(0, 0, 1))[0,0,::5])
    input_data = gaussian_filter(input_data, sigma=(0, 0, 1))[:, :, ::cfg['downsampling_step']]

    train_data = FullEEGDataset(np_input=input_data[indices[:split_idx], :, :],
                                np_targets=targets[indices[:split_idx]])
    val_data = FullEEGDataset(np_input=input_data[indices[split_idx:], :, :],
                              np_targets=targets[indices[split_idx:]])

    # data loaders
    train_loader = DataLoader(train_data, batch_size=cfg['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=False, num_workers=0)

    # model
    idx_hparams = {'input_channels': input_data.shape[1],
                   'n_states': len(np.unique(targets)),
                   'lr': cfg['lr'],
                   'epochs': cfg['epochs'],
                   'input_dropout': cfg['input_dropout'],
                   'num_classes': 3}

    model = LitConvClassifier(hparams=idx_hparams)

    for batch in train_loader:
        break

    batch=next(iter(train_loader))

    inputs, targets  = batch

    inputs.shape

    from torch.nn import functional as F
    import torch

    x = inputs.float().clone()
    self = model

    assert x.dim() == 3, f"Expected 3 dims: B, Electrodes, ms, got : {x.shape}"
    with torch.no_grad():
        # Pass the input tensor through each of our operations
        if self.input_dropout is not None:
            dp = nn.Dropout2d(p=self.input_dropout)
            x = dp(x.unsqueeze(-1))[:, :, :, 0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))

        # x.shape
        x = F.relu(self._global_avg_pooling_(x))
        x = self.fc2(x)


"""    def global_avg_pooling(self, x):
        return torch.mean(x, dim=2)"""

"""    output_dir = osp.join(cfg['experiments_dir'], f"conv_iter0_bs{cfg['batch_size']}")
    logger = TensorBoardLogger(save_dir=output_dir,
                               name=f"sub{subject}",
                               version=f"{datetime.now().strftime('%Y-%m-%d_%H%M')}")

    trainer = pl.Trainer(max_epochs=cfg['epochs'],
                         logger=logger)
    trainer.fit(model, train_loader, val_loader)"""
