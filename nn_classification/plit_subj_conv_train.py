import sys
sys.path.append('/Users/mcomastu/TFM/brain-states-dl') # TODO: remove this first two lines


import pytorch_lightning as pl
from torch.utils.data import DataLoader
from data_preprocessing.preprocess_module import sequence_downsampling
from nn_classification.data_loaders import FullEEGDataset, subject_nn_data
from nn_classification.pl_module import LitConvClassifier
from pytorch_lightning.loggers import TensorBoardLogger
from utils.file_utils import load_cfg
import numpy as np
from datetime import datetime
import os.path as osp
import torch


def main(cfg):
    # for subject in [25]:
    # cfg = load_cfg()
    for subject in cfg['healthy_subjects']:
        print('------------------------------------\nSubject', subject,
              '\n------------------------------------')
    # subject = 25
        input_data, targets, long_labels = subject_nn_data(subject,
                                                           healthy_subjects=cfg['healthy_subjects'],
                                                           pd_subjects=cfg['pd_subjects'],
                                                           feature_name=cfg['pred_feature'],
                                                           data_path=cfg['data_path'],
                                                           pd_dir=cfg['pd_dir'],
                                                           healthy_dir=cfg['healthy_dir'],
                                                           use_silent_channels=cfg['use_silent_channels'],
                                                           mask_value=cfg['mask_value'],
                                                           conv=True)

        # train-val split
        split_idx_path = osp.join(cfg['outputs_path'], cfg['splits_path'], f'{subject}-mlp.npy')

        if osp.exists(split_idx_path):
            indices = np.load(split_idx_path)
        else:
            indices = np.arange(input_data.shape[1])
            np.random.shuffle(indices)
            np.save(split_idx_path, indices)

        split_idx = int(input_data.shape[0] * 0.9)

        if cfg['sep_freqs']:
            freqs_idx = [0, 1, 2]
            freq_names = ['alpha', 'beta', 'gamma']
            # freqs_idx = [2]
            # freq_names = ['gamma']
        else:
            freqs_idx = [0]
            freq_names = [None]

        for freq_id in freqs_idx:
            if freq_names[0] is None:
                input_data_freq = input_data
            else:
                input_data_freq = input_data[:, freq_id * 60:freq_id * 60 + 60, :]

            freq_name = freq_names[freq_id]
            # freq_name = freq_names[0]

            input_data_freq = sequence_downsampling(input_data_freq, cfg['downsampling_step'])

            train_data = FullEEGDataset(np_input=input_data_freq[indices[:split_idx], :, :],
                                        np_targets=targets[indices[:split_idx]])
            val_data = FullEEGDataset(np_input=input_data_freq[indices[split_idx:], :, :],
                                      np_targets=targets[indices[split_idx:]])

            # data loaders
            train_loader = DataLoader(train_data, batch_size=cfg['batch_size'], shuffle=True, num_workers=0)
            val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=False, num_workers=0)

            # model
            idx_hparams = {'input_channels': input_data_freq.shape[1],
                           'kernel_size': 3,
                           'n_states': len(np.unique(targets)),
                           'lr': cfg['lr'],
                           'epochs': cfg['epochs'],
                           'input_dropout': cfg['input_dropout'],
                           'freq_name': freq_name,
                           'num_classes': 3}

            model = LitConvClassifier(hparams=idx_hparams)

            # training
            hparams_str = f"bs{cfg['batch_size']}_lr{cfg['lr']}"
            logger = TensorBoardLogger(save_dir=osp.join(cfg['experiments_dir'], f"subject-{subject}"),
                                       name=f"freq-{freq_name}-single_subject",
                                       version=f"CNN_{hparams_str}_"
                                               f"{datetime.now().strftime('%Y-%m-%d_%H%M')}")

            trainer = pl.Trainer(max_epochs=cfg['epochs'],
                                 logger=logger, deterministic=True)
            trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    torch.set_deterministic(True)
    np.random.seed(0)

    cfg = load_cfg()
    main(cfg)
