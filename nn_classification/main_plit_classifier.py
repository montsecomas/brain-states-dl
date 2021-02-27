import sys
sys.path.append('/Users/mcomastu/TFM/brain-states-dl') # TODO: remove this first two lines


import pytorch_lightning as pl
from torch.utils.data import DataLoader
from nn_classification.data_loaders import EEGDataset, subject_nn_data
from nn_classification.pl_module import LitClassifier
from pytorch_lightning.loggers import TensorBoardLogger
from utils.utils import load_cfg
import numpy as np
from datetime import datetime


if __name__ == '__main__':
    cfg = load_cfg()

    # for subject in [25]:
    for subject in cfg['healthy_subjects']:
        print('------------------------------------\nSubject', subject,
              '\n------------------------------------')
    # subject = 25
        input_data, targets = subject_nn_data(subject,
                                              healthy_subjects=cfg['healthy_subjects'],
                                              pd_subjects=cfg['pd_subjects'],
                                              feature_name=cfg['pred_feature'],
                                              data_path=cfg['data_path'],
                                              pd_dir=cfg['pd_dir'],
                                              healthy_dir=cfg['healthy_dir'],
                                              use_silent_channels=cfg['use_silent_channels'])

        freqs = ['alpha', 'beta', 'gamma']
        n_freqs = len(freqs)
        for freq in np.arange(n_freqs):
            # train-val split
            indices = np.arange(input_data.shape[1])
            np.random.shuffle(indices)
            split_idx = int(input_data.shape[1]*0.9)

            train_data = EEGDataset(np_input=input_data[freq, indices[:split_idx], :],
                                    np_targets=targets[indices[:split_idx]])
            val_data = EEGDataset(np_input=input_data[freq, indices[split_idx:], :],
                                  np_targets=targets[indices[split_idx:]])

            # data loaders
            train_loader = DataLoader(train_data, batch_size=cfg['batch_size'], shuffle=True, num_workers=0)
            val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=False, num_workers=0)

            # model
            idx_hparams = {'n_features': input_data.shape[2],
                           'n_states': len(np.unique(targets)),
                           'n_hidden_nodes': cfg['n_hidden_nodes'],
                           'n_hidden_layers': cfg['n_hidden_layers'],
                           'lr': cfg['lr']}
            model = LitClassifier(hparams=idx_hparams, freq_name=freqs[freq], pred_feature=cfg['pred_feature'],
                                  epochs=cfg['epochs'])

            # training
            prefix = 'POW_MEAN_' if (cfg['mat_dict'] == 'dataSorted') else 'IC_MEAN_'
            logger = TensorBoardLogger(save_dir=cfg['experiments_dir'],
                                       name=f"subject-{subject}-freq_{freqs[freq]}",
                                       version=f"{prefix}{datetime.now().strftime('%Y-%m-%d_%H%M')}")
            trainer = pl.Trainer(max_epochs=cfg['epochs'],
                                 logger=logger)
            trainer.fit(model, train_loader, val_loader)
