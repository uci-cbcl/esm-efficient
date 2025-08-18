import torch
import torch.nn.functional as F
import torchmetrics
from lightning.pytorch import LightningModule
from esme.loss import cross_entropy


class MaskedPLM(LightningModule):
    '''
    Trainer for masked language modeling.

    Args:
        model (torch.nn.Module): A model that predicts log probabilities of tokens.
        optimizer_name (str): Name of the optimizer to use.
        optimizer_kwargs (dict): Keyword arguments to pass to the optimizer.
    '''

    def __init__(self, model, optimizer_name='adam', optimizer_kwargs=None):
        super().__init__()
        self.model = model
        self.optimizer_name = optimizer_name
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.val_loss = torchmetrics.MeanMetric()

    def forward(self, tokens, pad_args):
        return self.model(tokens, pad_args)

    def _loss(self, target, pad_args, token, mask):
        return cross_entropy(self(token, pad_args), target, mask)

    def training_step(self, batch, batch_idx):
        target, pad_args, token, mask = batch
        loss = self._loss(target, pad_args, token, mask)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        target, pad_args, token, mask = batch
        loss = self._loss(target, pad_args, token, mask)
        self.val_loss.update(loss)
        self.log('val_loss', self.val_loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), **self.optimizer_kwargs)


class RegressionTrainer(LightningModule):

    def __init__(self, model, head, lr=1e-5, lr_head=1e-4, reduction='mean', layers=None):
        super().__init__()
        self.lr = lr
        self.lr_head = lr_head

        self.plm = model
        self.head = head
        self.reduction = reduction
        self.layers = layers

        self.val_loss = torchmetrics.MeanMetric()
        self.val_spearman = torchmetrics.SpearmanCorrCoef()

    def forward(self, token, pad_args, pad_output=False, pad_indices=None):

        embed = self.plm.forward_representation(
            token, pad_args, pad_output=pad_output, pad_indices=pad_indices, layers=self.layers)

        if self.reduction is None:
            return self.head(embed, pad_args)
        else:
            return self.head(self.reduce(embed))

    def reduce(self, x):
        if self.reduction == 'mean':
            return x.mean(dim=1)
        elif self.reduction == 'sum':
            return x.sum(dim=1)
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}, "
                             "expected `'mean'`, `'sum'` or `None`")

    def _loss(self, batch):
        y = self(batch['token'], (batch['cu_lens'], batch['max_len']),
                 pad_output=False, pad_indices=batch['indices'])
        labels = batch['label'].to(y.dtype)
        return F.mse_loss(y, labels, reduction='sum'), y

    def training_step(self, batch, batch_idx):
        loss, _ = self._loss(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y = self._loss(batch)
        self.val_loss.update(loss)
        labels = batch['label'].to(y.dtype)
        self.val_spearman.update(y, labels)
        self.log('val_loss', self.val_loss, on_epoch=True)
        self.log('val_spearman_corr', self.val_spearman, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam([
            {'params': self.head.parameters(), 'lr': self.lr_head},
            {'params': self.plm.parameters()}
        ], lr=self.lr)
