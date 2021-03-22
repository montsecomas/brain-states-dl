import sys
sys.path.append('/Users/mcomastu/TFM/brain-states-dl') # TODO: remove this first two lines


import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from nn_classification.data_loaders import FlatEEGDataset, subject_nn_data
from nn_classification.pl_module import LitConvClassifier
from pytorch_lightning.loggers import TensorBoardLogger
from utils.utils import load_cfg
import numpy as np
from datetime import datetime
import os.path as osp

TOTAL_ROWS = 13545
TOTAL_FEATURES = 180
TOTAL_MS = 1200


def main(cfg):
    # cfg = load_cfg()
    input_data = np.empty((TOTAL_ROWS, TOTAL_FEATURES, TOTAL_MS), dtype=np.float32)
    targets = np.empty((TOTAL_ROWS,), dtype=np.float32)
    subject_ids = np.empty((TOTAL_ROWS,), dtype=np.float32)
    last_row = 0
    for subject in cfg['healthy_subjects']:
    # for subject in [25, 26]:
        subject_input, subject_targets, long_labels = subject_nn_data(subject,
                                                                      healthy_subjects=cfg['healthy_subjects'],
                                                                      pd_subjects=cfg['pd_subjects'],
                                                                      feature_name=cfg['pred_feature'],
                                                                      data_path=cfg['data_path'],
                                                                      pd_dir=cfg['pd_dir'],
                                                                      healthy_dir=cfg['healthy_dir'],
                                                                      use_silent_channels=True,
                                                                      mask_value='0',
                                                                      conv=True)

        n_rows = subject_input.shape[0]
        input_data[last_row:  last_row + n_rows] = subject_input.astype(np.float32)
        targets[last_row:  last_row + n_rows] = subject_targets.astype(np.float32)
        subject_ids[last_row:  last_row + n_rows] = np.array(n_rows * [subject]).astype(np.float32)
        last_row += n_rows

    if last_row < TOTAL_ROWS:
        input_data = input_data[:last_row]
        targets = targets[:last_row]
        subject_ids = targets[:last_row]

    input_data = input_data[:, :, ::4]

    # for val_id in cfg['healthy_subjects']:
    for val_id in [25]:
        # val_id = 25
        print('------------------------------------\nCrossvalidation. Val with subject ', val_id,
              '\n------------------------------------')
        # train-val split
        val_mask = np.isin(subject_ids, val_id)
        val_inputs = input_data[val_mask]
        train_inputs = input_data[~val_mask]

        train_data = FlatEEGDataset(np_input=train_inputs,
                                    np_targets=targets[~val_mask])
        val_data = FlatEEGDataset(np_input=val_inputs,
                                  np_targets=targets[val_mask])

        # data loaders
        train_loader = DataLoader(train_data, batch_size=cfg['batch_size'], shuffle=True, num_workers=0)
        val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=False, num_workers=0)

        # model
        idx_hparams = {'input_channels': input_data.shape[1],
                       'kernel_size': 8,
                       'n_states': len(np.unique(targets)),
                       'lr': cfg['lr'],
                       'epochs': cfg['epochs'],
                       'input_dropout': None,
                       'num_classes': 3}

        model = LitConvClassifier(hparams=idx_hparams)

        # training
        output_dir = osp.join('exp_logs_conv_crossval', f"conv_v0_bs{cfg['batch_size']}")
        logger = TensorBoardLogger(save_dir=output_dir,
                                   name=f"sub{val_id}",
                                   version=f"{datetime.now().strftime('%Y-%m-%d_%H%M')}")

        trainer = pl.Trainer(max_epochs=cfg['epochs'],
                             logger=logger, deterministic=True)
        trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    torch.set_deterministic(True)
    np.random.seed(0)

    cfg = load_cfg()
    main(cfg)
