import torch
import torchmetrics
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from esme.loss import nll_loss


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
        return self.model.predict_log_prob(tokens, pad_args)

    def _loss(self, target, pad_args, token, mask):
        return nll_loss(self(token, pad_args), target, mask)

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
