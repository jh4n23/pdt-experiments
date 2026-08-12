import torch
import torch.nn as nn
from art.estimators.classification import PyTorchClassifier

from torch.utils.data import DataLoader
from base_classifier import BaseClassifier
from robust_classifier import RobustClassifier

class DataAugClassifier(BaseClassifier, RobustClassifier):
    def __init__(self):
        super().__init__()

    def train(self, train_loader: DataLoader, num_epochs: int, batch_size: int):
        # initial training stage to determine model weights for PGD attack
        BaseClassifier.train(train_loader, num_epochs, batch_size)

        # augment training data with adversarial examples
        x_batches, y_batches = [], []
        for images, labels in train_loader:
            x_batches.append(images)
            y_batches.append(labels)

        x_train = torch.cat(x_batches).numpy()
        y_train = torch.cat(y_batches).numpy()

        x_train_adv, y_train_adv = self.generate_adv_examples(
            x_train,
            y_train
        )

        # TODO: second training stage with clean + adv training data
        # can we reuse BaseClassifier.train() again?