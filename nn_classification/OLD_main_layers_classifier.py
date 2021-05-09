import sys
sys.path.append('/Users/mcomastu/TFM/brain-states-dl') # TODO: remove this first two lines


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils.file_utils import load_cfg
from nn_classification.models import TestNet
from nn_classification.data_loaders import FlatEEGDataset, subject_nn_data

READ_MODEL = False
MAKE_PRED = False


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
            bs = 64
            net = TestNet(n_features=input_data.shape[2], n_states=len(np.unique(targets)),
                          n_hidden_nodes_1=256, n_hidden_nodes_2=128)
            train_data = FlatEEGDataset(np_input=input_data[freq, :, :], np_targets=targets)
            trainloader = torch.utils.data.DataLoader(train_data,
                                                      batch_size=bs, shuffle=True, num_workers=0)
            # Define loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(net.parameters(), lr=0.001)
            # optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

            for epoch in range(5000):  # loop over the dataset multiple times

                running_loss = 0.0
                # data = next(iter(trainloader))
                # for i in range(10000):
                for i, data in enumerate(trainloader, 0):
                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = net(inputs.float())
                    loss = criterion(outputs, labels)
                    print(loss.item())
                    loss.backward()
                    optimizer.step()

            print('Finished Training')

            if READ_MODEL:
                PATH = './cifar_net.pth'
                torch.save(net.state_dict(), PATH)

            if MAKE_PRED:
                # Read saved network
                net = TestNet()
                net.load_state_dict(torch.load(PATH))
                outputs = net('input')
