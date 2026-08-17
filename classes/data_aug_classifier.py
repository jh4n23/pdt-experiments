import torch
from torch.utils.data import DataLoader, TensorDataset
from base_classifier import BaseClassifier
from robust_classifier import RobustClassifier

class DataAugClassifier(RobustClassifier, BaseClassifier):
    def __init__(self):
        super().__init__()

    def train(self, train_loader: DataLoader, num_epochs: int, batch_size: int):
        # initial training stage (3 epochs only) to determine model weights for PGD attack
        BaseClassifier.train(self, train_loader=train_loader, num_epochs=3, batch_size=batch_size)

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
        adv_dataset = TensorDataset(
            torch.from_numpy(x_train_adv),
            torch.from_numpy(y_train_adv)
        )
        adv_loader = DataLoader(adv_dataset, batch_size=batch_size, shuffle=True)
        BaseClassifier.train(self, train_loader=adv_loader, num_epochs=num_epochs, batch_size=batch_size)