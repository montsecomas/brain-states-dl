import torch.nn as nn
import torch.nn.functional as F


# Define the network
class TestNet(nn.Module):
    def __init__(self, n_features, n_states, n_hidden_nodes_1=128, n_hidden_nodes_2=64):
        super(TestNet, self).__init__()
        # Inputs to hidden layer linear transformation
        self.hidden_1 = nn.Linear(n_features, n_hidden_nodes_1)
        self.hidden_2 = nn.Linear(n_hidden_nodes_1, n_hidden_nodes_2)
        self.hidden_3 = nn.Linear(n_hidden_nodes_2, int(n_hidden_nodes_2/2))
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(int(n_hidden_nodes_2/2), n_states)

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
        x = self.hidden_3(x)
        x = self.relu(x)
        x = self.output(x)
        # x = self.softmax(x)
        return x


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, on_last=False):
        super().__init__()
        self.on_last = on_last
        self.dropout = dropout
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        if self.dropout is not None:
            for i, layer in enumerate(self.layers):
                x = F.relu(layer(F.dropout(x, self.dropout))) if i < self.num_layers - 1 or self.on_last else layer(x)
        else:
            for i, layer in enumerate(self.layers):
                x = F.relu(layer(x)) if i < self.num_layers - 1 or self.on_last else layer(x)
        return x
