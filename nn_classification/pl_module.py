import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from nn_classification.models import MLP
from sklearn.metrics import roc_auc_score


class LitClassifier(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        # Inputs to hidden layer linear transformation
        self.hparams = hparams
        self.mlp = MLP(input_dim=self.hparams['n_features'], hidden_dim=self.hparams['n_hidden_nodes'],
                       output_dim=self.hparams['n_states'], num_layers=self.hparams['n_hidden_layers'])

        # Define sigmoid activation and loss criterion
        self.relu = F.relu
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.mlp(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['lr'])
        return optimizer

    def log_metrics(self, metrics, trainval='train'):
        for metric_name, metric_val in metrics.items():
            self.log(f'{metric_name}/{trainval}', metric_val)

    def trainval_step(self, batch, batch_idx, trainval):
        inputs, labels = batch
        outputs = self(inputs.float())
        loss = self.criterion(outputs, labels)

        metrics = self.compute_metrics(outputs, labels)
        self.log_metrics({**metrics, 'loss': loss}, trainval=trainval)

        if trainval == 'train':
            return loss

    def training_step(self, batch, batch_idx):
        return self.trainval_step(batch, batch_idx, trainval='train')

    def validation_step(self, batch, batch_idx):
        return self.trainval_step(batch, batch_idx, trainval='val')

    @torch.no_grad()
    def compute_metrics(self, output, target):
        # TODO: parametrize num_classes/labels
        # auc = torch.as_tensor(roc_auc_score(F.one_hot(target, num_classes=3).numpy(), F.softmax(output, dim=-1).numpy(),
        #                                     average='macro', multi_class='ovr', labels=[0,1,2]))
        return {'acc': 0}

