import sys
sys.path.append('/Users/mcomastu/TFM/brain-states-dl') # TODO: remove this first two lines


import pytorch_lightning as pl
from torch.utils.data import DataLoader
from nn_classification.data_loaders import EEGDataset, subject_nn_data
from nn_classification.pl_module import LitClassifier
from utils.utils import load_cfg
import numpy as np


if __name__ == '__main__':
    cfg = load_cfg()

    # for subject in cfg['healthy_subjects']:
    for subject in [25]:
        # subject = 25
        input_data, targets = subject_nn_data(subject,
                                              healthy_subjects=cfg['healthy_subjects'],
                                              pd_subjects=cfg['pd_subjects'],
                                              feature_name='pow_mean',
                                              data_path=cfg['data_path'],
                                              pd_dir=cfg['pd_dir'],
                                              healthy_dir=cfg['healthy_dir'])

        n_freqs = 1
        for freq in np.arange(n_freqs):
            # freq = 2
            bs = 64
            # data
            train_data = EEGDataset(np_input=input_data[freq, :, :], np_targets=targets)
            # mnist_train, mnist_val = random_split(dataset, [55000, 5000])
            train_loader = DataLoader(train_data, batch_size=bs, shuffle=True, num_workers=0)
            val_loader = DataLoader(train_data, batch_size=bs, shuffle=True, num_workers=0)

            # model
            model = LitClassifier(n_features=input_data.shape[2],
                                  n_states=len(np.unique(targets)),
                                  n_hidden_nodes=128, n_hidden_layers=2, lr=1e-3)

            # training
            trainer = pl.Trainer(limit_train_batches=0.5, max_epochs=10000)
            trainer.fit(model, train_loader, val_loader)
