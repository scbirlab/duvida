"""Mixins for training."""

from lightning import LightningModule, Trainer
from torch.utils.data import DataLoader

from ...base.training import ModelTrainerBase
from ...stateless.config import config

config.set_backend('torch', precision='float')

from ...stateless.typing import Array

class ModelTrainer(ModelTrainerBase):

    _trainer: Trainer = None

    def create_trainer(self) -> None:
        self._trainer = Trainer(
            max_epochs=self.epochs,
            **self._kwargs,
        )
        return None

    def train(
        self, 
        model: LightningModule, 
        train_dataloader: DataLoader, 
        val_dataloader: DataLoader
    ) -> None:
        if self._trainer is None:
            self.create_trainer()
        self._trainer.fit(model, train_dataloader, val_dataloader)
        return model
