import sys
sys.path.append('/Users/mcomastu/TFM/brain-states-dl') # TODO: remove this first two lines


import pytorch_lightning as pl
from torch.utils.data import DataLoader
from nn_classification.data_loaders import FlatEEGDataset, subject_nn_data
from nn_classification.pl_module import LitMlpClassifier
from pytorch_lightning.loggers import TensorBoardLogger
from utils.file_utils import load_cfg
import numpy as np
from datetime import datetime
import os.path as osp
import os
import torch


def main(cfg):
    # for subject in [25]:
    # cfg = load_cfg()
    run_pd = cfg['run_pd']
    subjects_list = cfg['pd_subjects'] if run_pd else cfg['healthy_subjects']
    if run_pd:
        med_str = '-on-med' if cfg['model_on_med'] else '-off-med'
    else:
        med_str = ''

    for subject in subjects_list:
        print('------------------------------------\nSubject', subject,
              '\n------------------------------------')
        # subject = 25, healthy
        # subject = 55, PD
        input_data, targets, long_labels = subject_nn_data(subject,
                                                           healthy_subjects=cfg['healthy_subjects'],
                                                           pd_subjects=cfg['pd_subjects'],
                                                           feature_name=cfg['pred_feature'],
                                                           data_path=cfg['data_path'],
                                                           pd_dir=cfg['pd_dir'],
                                                           healthy_dir=cfg['healthy_dir'],
                                                           on_med=cfg['model_on_med'],
                                                           use_silent_channels=cfg['use_silent_channels'],
                                                           mask_value=cfg['mask_value'])

        freqs = ['alpha', 'beta', 'gamma']
        freqs_idx = [0, 1, 2]
        # freq = 0

        split_idx_path = osp.join(cfg['outputs_path'], cfg['splits_path'], f'{subject}{med_str}-mlp.npy')

        if osp.exists(split_idx_path):
            indices = np.load(split_idx_path)
        else:
            indices = np.arange(input_data.shape[1])
            np.random.shuffle(indices)
            np.save(split_idx_path, indices)

        split_idx = int(input_data.shape[1] * 0.9)

        for freq in freqs_idx:
            # train-val split

            train_data = FlatEEGDataset(np_input=input_data[freq, indices[:split_idx], :],
                                        np_targets=targets[indices[:split_idx]])
            val_data = FlatEEGDataset(np_input=input_data[freq, indices[split_idx:], :],
                                      np_targets=targets[indices[split_idx:]])

            # data loaders
            train_loader = DataLoader(train_data, batch_size=cfg['batch_size'], shuffle=True, num_workers=0)
            val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=False, num_workers=0)

            # model
            idx_hparams = {'n_features': input_data.shape[2],
                           'n_states': len(np.unique(targets)),
                           'n_hidden_nodes': cfg['n_hidden_nodes'],
                           'n_hidden_layers': cfg['n_hidden_layers'],
                           'lr': cfg['lr'],
                           'epochs': cfg['epochs'],
                           'freq_name': freqs[freq],
                           'pred_feature': cfg['pred_feature'],
                           'input_dropout': cfg['input_dropout'],
                           'mlp_dropout': cfg['mlp_dropout'],
                           'weight_decay': cfg['weight_decay'],
                           'num_classes': 3}

            model = LitMlpClassifier(hparams=idx_hparams)

            # training
            prefix = 'pow-mean' if (cfg['mat_dict'] == 'dataSorted') else 'IC-MEAN'
            logger = TensorBoardLogger(save_dir=osp.join(cfg['experiments_dir'], f"subject-{subject}"),
                                       name=f"freq-{freqs[freq]}-single_subject",
                                       version=f"MLP{med_str}-{prefix}_{datetime.now().strftime('%Y-%m-%d_%H%M')}")
            trainer = pl.Trainer(max_epochs=cfg['epochs'],
                                 logger=logger)
            trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    torch.set_deterministic(True)
    np.random.seed(0)

    cfg = load_cfg()
    main(cfg)
