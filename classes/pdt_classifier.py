import torch
import os
import torch.nn as nn
import vehicle_lang as vcl
from base_classifier import BaseClassifier
from vehicle_lang.loss import pytorch as loss_pt
from torch.utils.data import DataLoader

"""
Standard PDT!
"""

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
SPEC_PATH = os.path.join(_MODULE_DIR, "specs", "mnist-robustness.vcl")

class PdtClassifier(BaseClassifier):
    def __init__(self):
        super().__init__()

    def grad_norm(self, loss, params):
        grads = torch.autograd.grad(loss, params, retain_graph=True, create_graph=False, allow_unused=True)
        grads = [g for g in grads if g is not None] # redundant?
        if len(grads) == 0:
            return torch.tensor(0.0, device=params[0].device)
        
        return torch.sqrt(sum((g ** 2).sum() for g in grads))

    @staticmethod
    def get_spec():
        return loss_pt.load_specification(
            SPEC_PATH,
            logic=vcl.VehicleDifferentiableLogic()
        )

    def network(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x.reshape(1, 1, 28, 28)).reshape(10)

    """ Compute task vs constraint weighting parameter """
    def compute_lam(self, task_loss, constraint_loss):
        params = [p for p in self.model.parameters() if p.requires_grad]
        task_grad_norm = self.grad_norm(task_loss, params)
        constraint_grad_norm = self.grad_norm(constraint_loss, params)
        lam = (task_grad_norm / (task_grad_norm + constraint_grad_norm + 1e-8)).item()
        return lam

    def train(self, train_loader: DataLoader, num_epochs: int, batch_size: int):
        constraint_loss_fn = self.get_spec()["robust"]

        criterion = nn.CrossEntropyLoss()
        for epoch in range(num_epochs):
            lam = None
            for images, labels in train_loader:
                self.optimizer.zero_grad()
                logits = self.model(images)
                task_loss = criterion(logits, labels)

                constraint_loss = torch.stack(constraint_loss_fn(
                    n=images.shape[0],
                    classifier=self.network,
                    epsilon=torch.tensor(0.005),
                    trainingImages=images.squeeze(1),
                    trainingLabels=labels
                )).mean()

                if lam is None:
                    lam = self.compute_lam(task_loss, constraint_loss)

                total_loss = (1 - lam) * task_loss + lam * constraint_loss
                total_loss.backward()
                self.optimizer.step()