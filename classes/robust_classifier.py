from base_classifier import BaseClassifier
from torch.utils.data import DataLoader
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import ProjectedGradientDescent

import torch
import torch.nn as nn
import numpy as np

class RobustClassifier(BaseClassifier):
    def __init__(self):
        super().__init__()

    def build_attack(self):
        criterion = nn.CrossEntropyLoss()
        classifier = PyTorchClassifier(
            model=self.model, loss=criterion, optimizer=self.optimizer,
            input_shape=(1, 28, 28), nb_classes=10,
        )
        return ProjectedGradientDescent(estimator=classifier, eps=0.1, eps_step=0.01, max_iter=10, verbose=False)

    def generate_adv_examples(self, x_batch, y_batch):
        attack = self.build_attack()
        x_batch_adv = attack.generate(x=x_batch)

        x_combined = np.concatenate([x_batch, x_batch_adv], axis=0)
        y_combined = np.concatenate([y_batch, y_batch], axis=0)

        perm = np.random.permutation(len(x_combined))
        x_shuffled = x_combined[perm]
        y_shuffled = y_combined[perm]
        return x_shuffled, y_shuffled

    def train(self, train_loader: DataLoader, num_epochs: int, batch_size: int):
        x_batches, y_batches = [], []
        for images, labels in train_loader:
            x_batches.append(images)
            y_batches.append(labels)

        x_train = torch.cat(x_batches).numpy()
        y_train = torch.cat(y_batches).numpy()

        criterion = nn.CrossEntropyLoss()

        classifier = PyTorchClassifier(
            model=self.model,
            loss=criterion,
            optimizer=self.optimizer,
            input_shape=(1, 28, 28),
            nb_classes=10
        )

        for _ in range(num_epochs):
            for start in range (0, len(x_train), batch_size):
                end = start + batch_size
                x_batch = x_train[start:end]
                y_batch = y_train[start:end]

                x_batch_adv, y_batch_adv = self.generate_adv_examples(x_batch, y_batch)

                classifier.fit(
                    x_batch_adv,
                    y_batch_adv,
                    batch_size=len(x_batch_adv),
                    nb_epochs=1,
                )