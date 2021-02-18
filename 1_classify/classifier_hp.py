import sys
sys.path.append('/Users/mcomastu/TFM/brain-states-dl') # TODO: remove this first two lines


import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils.utils import load_cfg, is_pd_patient, processed_data_path, processed_labels_path
from sklearn import preprocessing

READ_MODEL = False
MAKE_PRED = False


# Define the network
class Net(nn.Module):
    def __init__(self, n_features, n_states, n_hidden_nodes_1=128, n_hidden_nodes_2=64):
        super(Net, self).__init__()
        # Inputs to hidden layer linear transformation
        self.hidden_1 = nn.Linear(n_features, n_hidden_nodes_1)
        self.hidden_2 = nn.Linear(n_hidden_nodes_1, n_hidden_nodes_2)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(n_hidden_nodes_2, n_states)

        # Define sigmoid activation and softmax output
        # self.sigmoid = nn.Sigmoid()
        self.relu = F.relu
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden_1(x)
        x = self.relu(x)
        x = self.hidden_2(x)
        x = self.relu(x)
        x = self.output(x)
        # x = self.softmax(x)
        return x


class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, np_input, np_targets):
        self.np_input = torch.from_numpy(np_input)
        self.np_targets = torch.from_numpy(np_targets)

    def __len__(self):
        return self.np_input.shape[0]

    def __getitem__(self, idx):
        return self.np_input[idx], self.np_targets[idx]


if __name__ == '__main__':
    cfg = load_cfg()

    for subject in cfg['healthy_subjects']:
        subject = 25
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

        # TODO: test the following
        n_freqs = 1
        for freq in np.arange(n_freqs):
            net = Net(n_features=input_data.shape[2], n_states=len(np.unique(np_labels[:, 1])),
                      n_hidden_nodes_1=128, n_hidden_nodes_2=64)
            train_data = EEGDataset(np_input=input_data[freq, :, :], np_targets=targets)
            trainloader = torch.utils.data.DataLoader(train_data,
                                                      batch_size=16, shuffle=True, num_workers=0)
            # Define loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

            for epoch in range(10):  # loop over the dataset multiple times

                running_loss = 0.0
                # for i, data in enumerate(trainloader, 0):
                data = next(iter(trainloader))
                for i in range(10000):
                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = net(inputs.float())
                    # outputs = torch.rand((16, 3))
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # print statistics
                    running_loss += loss.item()
                    if i % 1 == 0:    # print every 2000 mini-batches
                        print('[%d, %5d] loss: %.6f' %
                              (epoch + 1, i + 1, running_loss/1))
                        running_loss = 0.0

            print('Finished Training')

            if READ_MODEL:
                PATH = './cifar_net.pth'
                torch.save(net.state_dict(), PATH)

            if MAKE_PRED:
                # Read saved network
                net = Net()
                net.load_state_dict(torch.load(PATH))
                outputs = net('input')
