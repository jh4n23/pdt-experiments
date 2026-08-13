import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils.cnn import CNN
from torch.utils.data import DataLoader
from utils.csec_tester import CsecTester
from art.estimators.classification import PyTorchClassifier

class BaseClassifier():
    def __init__(self):
        self.model = CNN(in_channels=1, input_size=28, num_classes=10)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def train(self, train_loader: DataLoader, num_epochs: int, batch_size: int):
        criterion = nn.CrossEntropyLoss()
        for _ in range(num_epochs):
            for images, labels in train_loader:
                self.optimizer.zero_grad()
                logits = self.model(images)
                loss = criterion(logits, labels)

                loss.backward()
                self.optimizer.step()

    def evaluate(self, test_data):
        classifier = PyTorchClassifier(
            model=self.model,
            loss=nn.CrossEntropyLoss(),
            optimizer=self.optimizer,
            input_shape=(1, 28, 28),
            nb_classes=10,
        )

        x_test = torch.stack([img for img, _ in test_data]).numpy()
        y_test = torch.tensor([label for _, label in test_data]).numpy()

        preds = classifier.predict(x_test)
        accuracy = np.sum(np.argmax(preds, axis=1) == y_test) / len(y_test)
        tester = CsecTester(model=classifier)
        adv_accuracy = tester.run(x_test, y_test, 0.1)

        print(f"{self.__class__.__name__}:"
              f"\n\tBenign accuracy: {accuracy * 100:.1f}%"
              f"\n\tConstraint security: {adv_accuracy * 100:.1f}%"
        )

    def export(self, path):
        input = torch.randn(1, 1, 28, 28)
        torch.onnx.export(
            self.model,
            input,
            path,
            external_data=False
        )
        print(f"Saved {self.__class__.__name__} to {path}")