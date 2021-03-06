import sys
sys.path.append('/Users/mcomastu/TFM/brain-states-dl') # TODO: remove this first two lines


import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from nn_classification.data_loaders import FlatEEGDataset, subject_nn_data
from nn_classification.pl_module import LitMlpClassifier
from pytorch_lightning.loggers import TensorBoardLogger
from utils.file_utils import load_cfg
import numpy as np
from datetime import datetime
import os.path as osp


def main(cfg):
    all_targets = []
    all_inputs = []
    subject_ids = []
    for subject in cfg['healthy_subjects']:
        subject_input, subject_targets, long_labels = subject_nn_data(subject,
                                                                      healthy_subjects=cfg['healthy_subjects'],
                                                                      pd_subjects=cfg['pd_subjects'],
                                                                      feature_name=cfg['pred_feature'],
                                                                      data_path=cfg['data_path'],
                                                                      pd_dir=cfg['pd_dir'],
                                                                      healthy_dir=cfg['healthy_dir'],
                                                                      use_silent_channels=cfg['use_silent_channels'],
                                                                      mask_value=cfg['mask_value'])

        all_inputs.append(subject_input)
        all_targets.append(subject_targets)
        subject_ids.append(np.array(subject_input.shape[1] * [subject]))

    input_data = np.concatenate(all_inputs, axis=1)
    targets = np.concatenate(all_targets)
    subject_ids = np.concatenate(subject_ids)

    for val_id in cfg['healthy_subjects']:
        # val_id = 25
        # for val_id in cfg['healthy_subjects']:
        print('------------------------------------\nCrossvalidation. Val with subject ', val_id,
              '\n------------------------------------')
        # train-val split
        val_mask = np.isin(subject_ids, val_id)
        train_inputs = input_data[:, ~val_mask]
        val_inputs = input_data[:, val_mask]

        freqs = ['alpha', 'beta', 'gamma']
        freqs_idx = [0, 1, 2]
        # freq = 0
        for freq in freqs_idx:
            train_data = FlatEEGDataset(np_input=train_inputs[freq, :, :],
                                        np_targets=targets[~val_mask])
            val_data = FlatEEGDataset(np_input=val_inputs[freq, :, :],
                                      np_targets=targets[val_mask])

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
            prefix = 'pow-mean'
            logger = TensorBoardLogger(save_dir=osp.join(cfg['experiments_dir'], f"subject-{val_id}"),
                                       name=f"freq-{freqs[freq]}-crossval",
                                       version=f"MLP_{datetime.now().strftime('%Y-%m-%d_%H%M')}")

            trainer = pl.Trainer(max_epochs=cfg['epochs'],
                                 logger=logger)
            trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    torch.set_deterministic(True)
    np.random.seed(0)

    cfg = load_cfg()
    main(cfg)
