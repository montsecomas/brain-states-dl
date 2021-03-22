import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from nn_classification.models import MLP
from sklearn.metrics import roc_auc_score
from pytorch_lightning.metrics.classification.confusion_matrix import ConfusionMatrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.autograd import Variable


class LitMlpClassifier(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        # Inputs to hidden layer linear transformation
        self.hparams = hparams
        self.num_classes = self.hparams['num_classes']
        self.epochs = self.hparams['epochs']
        self.freq_name = self.hparams['freq_name']
        self.pred_feature = self.hparams['pred_feature']
        self.weight_decay = self.hparams['weight_decay'] if self.hparams['weight_decay'] is not None else 0
        self.input_dropout = self.hparams['input_dropout']
        self.mlp_dropout = self.hparams['mlp_dropout']
        self.feature = 'pow' if self.pred_feature == 'pow_mean' else 'ic'
        self.mlp = MLP(input_dim=self.hparams['n_features'], hidden_dim=self.hparams['n_hidden_nodes'],
                       output_dim=self.hparams['n_states'], num_layers=self.hparams['n_hidden_layers'],
                       dropout=self.mlp_dropout)

        # Define sigmoid activation and loss criterion
        self.relu = F.relu
        self.criterion = nn.CrossEntropyLoss()
        if self.freq_name == 'alpha':
            self.cmap = 'Greens'
        elif self.freq_name == 'beta':
            self.cmap = 'Blues'
        elif self.freq_name == 'gamma':
            self.cmap = 'Reds'

    def forward(self, x):
        # Pass the input tensor through each of our operations
        if self.input_dropout is not None:
            x = F.dropout(x, self.input_dropout)
        x = self.mlp(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['lr'], weight_decay=self.weight_decay)
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
        # self.log('Confusion matrix', ConfusionMatrix(F.softmax(output, dim=-1), labels, num_classes = 3))

        if trainval == 'train':
            return loss
        else:
            return {'loss': loss, 'preds': outputs, 'target': labels}

    def training_step(self, batch, batch_idx):
        return self.trainval_step(batch, batch_idx, trainval='train')

    def validation_step(self, batch, batch_idx):
        return self.trainval_step(batch, batch_idx, trainval='val')

    @torch.no_grad()
    def compute_metrics(self, output, target):
        # TODO: parametrize num_classes/labels
        auc = torch.as_tensor(roc_auc_score(target.numpy(), F.softmax(output, dim=-1).numpy(),
                                            average='macro', multi_class='ovo', labels=[0, 1, 2]))
        return {'auc': auc}

    def validation_epoch_end(self, outputs):
        if self.current_epoch in map(int, [self.epochs/self.epochs, self.epochs/(self.epochs/2), self.epochs-1]):
            preds = torch.cat([tmp['preds'] for tmp in outputs])
            targets = torch.cat([tmp['target'] for tmp in outputs])
            cf = ConfusionMatrix(num_classes=self.num_classes, normalize='true')
            confusion_matrix = cf(F.softmax(preds, dim=-1), targets)

            df_cm = pd.DataFrame(confusion_matrix.numpy(), index=range(self.num_classes),
                                 columns=range(self.num_classes))
            plt.figure(figsize=(25, 18))
            plt.rcParams['font.size'] = 40
            fig_ = sns.heatmap(df_cm, annot=True, cmap=self.cmap).get_figure()
            plt.title(f'{self.freq_name}, {self.feature}', fontsize=50)
            plt.xlabel('Predicted label', fontsize=50)
            plt.ylabel('True label', fontsize=50)
            plt.close(fig_)

            self.logger.experiment.add_figure("Confusion matrix", fig_, self.current_epoch)


class LitConvClassifier(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        # Inputs to hidden layer linear transformation
        self.hparams = hparams
        self.input_channels = self.hparams['input_channels']
        self.kernel_size = self.hparams['kernel_size']
        self.n_states = self.hparams['n_states']
        self.lr = self.hparams['lr']
        self.epochs = self.hparams['epochs']
        self.num_classes = self.hparams['num_classes']
        self.input_dropout = self.hparams['input_dropout']

        self.conv1 = nn.Conv1d(self.input_channels, 512, self.kernel_size, stride=2, padding=1)
        self.conv2 = nn.Conv1d(512, 256, self.kernel_size, stride=2, padding=1)
        self.conv3 = nn.Conv1d(256, 128, self.kernel_size, stride=2, padding=1)

        # self.fc1 = nn.AdaptiveAvgPool1d(128)
        self.fc2 = nn.Linear(128, self.hparams['n_states'])

        self.criterion = nn.CrossEntropyLoss()

    def global_avg_pooling(self, x):
        return torch.mean(x, dim=2)

    def forward(self, x):
        # Pass the input tensor through each of our operations
        if self.input_dropout is not None:
            x = F.dropout(x, self.input_dropout)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.global_avg_pooling(x))
        x = self.fc2(x)
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
        # self.log('Confusion matrix', ConfusionMatrix(F.softmax(output, dim=-1), labels, num_classes = 3))

        if trainval == 'train':
            return loss
        else:
            return {'loss': loss, 'preds': outputs, 'target': labels}

    def training_step(self, batch, batch_idx):
        return self.trainval_step(batch, batch_idx, trainval='train')

    def validation_step(self, batch, batch_idx):
        return self.trainval_step(batch, batch_idx, trainval='val')

    @torch.no_grad()
    def compute_metrics(self, output, target):
        # TODO: parametrize num_classes/labels
        auc = torch.as_tensor(roc_auc_score(target.numpy(), F.softmax(output, dim=-1).numpy(),
                                            average='macro', multi_class='ovo', labels=[0, 1, 2]))
        return {'auc': auc}

    def validation_epoch_end(self, outputs):
        if self.current_epoch in map(int, [self.epochs/self.epochs, self.epochs/(self.epochs/2), self.epochs-1]):
            preds = torch.cat([tmp['preds'] for tmp in outputs])
            targets = torch.cat([tmp['target'] for tmp in outputs])
            cf = ConfusionMatrix(num_classes=self.num_classes, normalize='true')
            confusion_matrix = cf(F.softmax(preds, dim=-1), targets)

            df_cm = pd.DataFrame(confusion_matrix.numpy(), index=range(self.num_classes),
                                 columns=range(self.num_classes))
            plt.figure(figsize=(25, 18))
            plt.rcParams['font.size'] = 40
            fig_ = sns.heatmap(df_cm, annot=True).get_figure()
            plt.xlabel('Predicted label', fontsize=50)
            plt.ylabel('True label', fontsize=50)
            plt.close(fig_)

            self.logger.experiment.add_figure("Confusion matrix", fig_, self.current_epoch)



