import sys
sys.path.append('/Users/mcomastu/TFM/brain-states-dl') # TODO: remove this first two lines


import argparse
from torch.utils.data import DataLoader
from nn_classification.data_loaders import EEGDataset, subject_nn_data
from nn_classification.pl_module import LitClassifier
from utils.utils import load_cfg
import numpy as np
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing script')
    parser.add_argument('--test_subject', default=25)
    parser.add_argument('--ckpt_path', default=None)

    try:
        args = parser.parse_args()
    except:
        args = parser.parse_args([])

    cfg = load_cfg()
    test_subject = args.test_subject
    # test_subject = 25
    print('------------------------------------\nTesting subject', test_subject,
          '\n------------------------------------')
    input_data, targets, long_labels = subject_nn_data(test_subject,
                                                       healthy_subjects=cfg['healthy_subjects'],
                                                       pd_subjects=cfg['pd_subjects'],
                                                       feature_name=cfg['pred_feature'],
                                                       data_path=cfg['data_path'],
                                                       pd_dir=cfg['pd_dir'],
                                                       healthy_dir=cfg['healthy_dir'],
                                                       use_silent_channels=cfg['use_silent_channels'],
                                                       mask_value=cfg['mask_value'])

    freqs = ['gamma']
    n_freqs = len(freqs)
    for freq in np.arange(n_freqs):
        test_data = EEGDataset(np_input=input_data[freq, :, :], np_targets=targets)

        # data loader
        val_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False, num_workers=0)

        ###
        #args.ckpt_path= 'mlp_lightning_logs/subject-25-freq_gamma/POW-MEAN_2021-02-28_2014_ALL-CHANNELS_MASK-mean/checkpoints/epoch=499-step=9499.ckpt'
        ckpt = torch.load(args.ckpt_path)
        ckpt.keys()
        model = LitClassifier(hparams=ckpt['hyper_parameters'], freq_name=freqs[freq],
                              pred_feature=cfg['pred_feature'], epochs=cfg['epochs'])

        model.load_state_dict(ckpt['state_dict'])
        model.eval()

        batch = next(iter(val_loader))
        inputs, targets = batch
        with torch.no_grad():
            preds = model(inputs.float())

        # list_of_samples= [test_data[ix] for ix in [1, 3, 5, 2]]
        # batch = val_loader.collate_fn(list_of_samples)
        # inputs, targets = batch
