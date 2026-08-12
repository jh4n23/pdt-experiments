import torch
import torch.nn as nn
import vehicle_lang as vcl
from torch.utils.data import DataLoader
from pdt_classifier import PdtClassifier
from robust_classifier import RobustClassifier


class RobustPdtClassifier(PdtClassifier, RobustClassifier):
    def __init__(self):
        super().__init__()

    def train(self, train_loader: DataLoader, num_epochs: int, batch_size: int):
        x_batches, y_batches = [], []
        for images, labels in train_loader:
            x_batches.append(images)
            y_batches.append(labels)
        x_train = torch.cat(x_batches).numpy()
        y_train = torch.cat(y_batches).numpy()

        constraint_loss_fn = self.get_spec()["robust"]
        criterion = nn.CrossEntropyLoss()

        lam = None
        for _ in range(num_epochs):
            for start in range(0, len(x_train), batch_size):
                end = start + batch_size
                x_batch = x_train[start:end]
                y_batch = y_train[start:end]

                x_batch_adv, y_batch_adv = self.generate_adv_examples(x_batch, y_batch)

                images = torch.tensor(x_batch_adv, dtype=torch.float32)
                labels = torch.tensor(y_batch_adv, dtype=torch.long)

                self.optimizer.zero_grad()
                logits = self.model(images)
                task_loss = criterion(logits, labels)

                constraint_loss = torch.stack(constraint_loss_fn(
                    n=images.shape[0],
                    classifier=self.network,
                    epsilon=torch.tensor(0.005),
                    trainingImages=images.squeeze(1),
                    trainingLabels=labels,
                )).mean()

                if lam is None:
                    lam = self.compute_lam(task_loss, constraint_loss)

                total_loss = (1 - lam) * task_loss + lam * constraint_loss
                total_loss.backward()
                self.optimizer.step()