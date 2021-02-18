import sys
sys.path.append('/Users/mcomastu/TFM/brain-states-dl') # TODO: remove this first two lines


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils.utils import load_cfg, is_pd_patient, processed_data_path, processed_labels_path
from sklearn import preprocessing
from models import TestNet, MLP
from data_loaders import EEGDataset

READ_MODEL = False
MAKE_PRED = False


if __name__ == '__main__':
    cfg = load_cfg()

    # for subject in cfg['healthy_subjects']:
    for subject in [25]:
        # subject = 25
        is_pd = is_pd_patient(subject, cfg['healthy_subjects'], cfg['pd_subjects'])
        input_data = np.load(processed_data_path(subject_id=subject, is_pd=is_pd, feature_name='pow_mean',
                                                 data_path=cfg['data_path'], pd_dir=cfg['pd_dir'],
                                                 healthy_dir=cfg['healthy_dir']))
        np_labels = np.load(processed_labels_path(subject_id=subject, is_pd=is_pd, data_path=cfg['data_path'],
                                                  pd_dir=cfg['pd_dir'], healthy_dir=cfg['healthy_dir']))
        le = preprocessing.LabelEncoder()
        le.fit(np_labels[:, 1])
        # le.classes_
        targets = le.transform(np_labels[:, 1])

        n_freqs = 1
        for freq in np.arange(n_freqs):
            bs = 64
            net = TestNet(n_features=input_data.shape[2], n_states=len(np.unique(np_labels[:, 1])),
                          n_hidden_nodes_1=256, n_hidden_nodes_2=128)
            train_data = EEGDataset(np_input=input_data[freq, :, :], np_targets=targets)
            trainloader = torch.utils.data.DataLoader(train_data,
                                                      batch_size=bs, shuffle=True, num_workers=0)
            # Define loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(net.parameters(), lr=0.001)
            # optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

            for epoch in range(1000):  # loop over the dataset multiple times

                running_loss = 0.0
                # data = next(iter(trainloader))
                # for i in range(10000):
                for i, data in enumerate(trainloader, 0):
                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data
                    print(i)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = net(inputs.float())
                    # outputs = torch.rand((16, 3))
                    loss = criterion(outputs, labels)
                    print(loss.item())
                    loss.backward()
                    optimizer.step()

                    # print statistics
                    running_loss += loss.item()
                    if i % bs == bs-1:    # print every mini-batches
                        print('[%d, %5d] loss: %.6f' %
                              (epoch + 1, i + 1, running_loss/bs))
                        running_loss = 0.0

            print('Finished Training')

            if READ_MODEL:
                PATH = './cifar_net.pth'
                torch.save(net.state_dict(), PATH)

            if MAKE_PRED:
                # Read saved network
                net = TestNet()
                net.load_state_dict(torch.load(PATH))
                outputs = net('input')
