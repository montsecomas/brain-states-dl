import sys
sys.path.append('/Users/mcomastu/TFM/brain-states-dl') # TODO: remove this first two lines


import pytorch_lightning as pl
from torch.utils.data import DataLoader
from data_preprocessing.preprocess_module import sequence_downsampling
from nn_classification.data_loaders import FullEEGDataset, subject_nn_data
from nn_classification.pl_module import LitConvClassifier
from pytorch_lightning.loggers import TensorBoardLogger
from utils.utils import load_cfg
import numpy as np
from datetime import datetime
import os.path as osp
import torch


def main(cfg):
    for subject in [25]:
    # for subject in cfg['healthy_subjects']:
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
                                                           use_silent_channels=True,
                                                           mask_value='0',
                                                           conv=True)

        # train-val split
        indices = np.arange(input_data.shape[0])
        np.random.shuffle(indices)
        split_idx = int(input_data.shape[0]*0.9)

        if cfg['gamma_freq']:
            input_data = input_data[:, 120:, :]
            freq_prefix = 'freq-gamma'
            freq_name = 'gamma'
        else:
            freq_prefix = ''
            freq_name = None

        # ts downsampling
        # import matplotlib.pyplot as plt
        # plt.plot(np.arange(1200), input_data[0,0,:])
        # plt.plot(np.arange(1200)[::5], input_data[0,0,::5])
        # plt.plot(np.arange(1200), gaussian_filter(input_data, sigma=1)[0,0,:])
        # plt.plot(np.arange(1200)[::5], gaussian_filter(input_data, sigma=1)[0,0,::5])
        input_data = sequence_downsampling(input_data, cfg['downsampling_step'])

        train_data = FullEEGDataset(np_input=input_data[indices[:split_idx], :, :],
                                    np_targets=targets[indices[:split_idx]])
        val_data = FullEEGDataset(np_input=input_data[indices[split_idx:], :, :],
                                  np_targets=targets[indices[split_idx:]])

        # data loaders
        train_loader = DataLoader(train_data, batch_size=cfg['batch_size'], shuffle=True, num_workers=0)
        val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=False, num_workers=0)

        # model
        idx_hparams = {'input_channels': input_data.shape[1],
                       'kernel_size': 3,
                       'n_states': len(np.unique(targets)),
                       'lr': cfg['lr'],
                       'epochs': cfg['epochs'],
                       'input_dropout': cfg['input_dropout'],
                       'freq_name': freq_name,
                       'num_classes': 3}

        model = LitConvClassifier(hparams=idx_hparams)

        # training
        logger = TensorBoardLogger(save_dir=osp.join(cfg['experiments_dir'], f"subject-{subject}"),
                                   name=f"{freq_prefix}-single_subject",
                                   version=f"CNN_pyDropout_{datetime.now().strftime('%Y-%m-%d_%H%M')}")

        trainer = pl.Trainer(max_epochs=cfg['epochs'],
                             logger=logger, deterministic=True)
        trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    torch.set_deterministic(True)
    np.random.seed(0)

    cfg = load_cfg()
    main(cfg)
