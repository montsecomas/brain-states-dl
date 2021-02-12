import sys
sys.path.append('/Users/Meme/TFM/source')


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils.utils import load_cfg, is_pd_patient, processed_data_path

READ_MODEL = False
MAKE_PRED = False


# Define the network
class Net(nn.Module):
    def __init__(self, n_features, n_states):
        super(Net, self).__init__()
        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(n_features, 256)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(256, n_states)

        # Define sigmoid activation and softmax output
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)
        return x


if __name__ == '__main__':
    cfg = load_cfg()

    for subject in cfg['healthy_subjects']:
        subject = 25
        is_pd = is_pd_patient(subject, cfg['healthy_subjects'], cfg['pd_subjects'])
        input_data = np.load(processed_data_path(subject_id=subject, is_pd=is_pd, feature_name='pow_mean',
                                                 data_path=cfg['data_path'], pd_dir=cfg['pd_dir'],
                                                 healthy_dir=cfg['healthy_dir']))
        for freq in np.arange(freq):
            freq = 0
            net = Net(n_features=input_data.shape[2], n_states=6)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                                      shuffle=True, num_workers=2)
            # Define loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

            for epoch in range(2):  # loop over the dataset multiple times

                running_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # print statistics
                    running_loss += loss.item()
                    if i % 2000 == 1999:    # print every 2000 mini-batches
                        print('[%d, %5d] loss: %.3f' %
                              (epoch + 1, i + 1, running_loss / 2000))
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
