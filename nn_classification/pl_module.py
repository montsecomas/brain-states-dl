import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from models import TestNet, MLP


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, n_features, n_states, n_hidden_nodes=128, n_hidden_layers=2, lr=1e-3):
        super().__init__()
        # Inputs to hidden layer linear transformation
        self.mlp = MLP(input_dim=n_features, hidden_dim=n_hidden_nodes, output_dim=n_states, num_layers=n_hidden_layers)

        # Define sigmoid activation and softmax output
        self.relu = F.relu
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.mlp(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        inputs, labels = train_batch
        outputs = self(inputs.float())
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('val_loss', loss)

