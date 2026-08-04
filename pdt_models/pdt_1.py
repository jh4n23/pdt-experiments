import torch
import onnx
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import vehicle_lang as vcl

from torch.utils.data import DataLoader, TensorDataset
from vehicle_lang.loss import pytorch as loss_pt
from art.estimators.classification import PyTorchClassifier
from art.utils import load_mnist
from utils.cnn import CNN
from utils.csec_tester import CsecTester
from utils.onnx_exporter import OnnxExporter

BATCH_SIZE = 64
SUBSET_SIZE = 4096
NUM_EPOCHS = 10

(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()

x_train = x_train[:SUBSET_SIZE]
y_train = y_train[:SUBSET_SIZE]

x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)

model = CNN(in_channels=1, input_size=28, num_classes=10)

def network(x: torch.Tensor) -> torch.Tensor:
    return model(x.reshape(1, 1, 28, 28)).reshape(10)

def grad_norm(loss, params):
    grads = torch.autograd.grad(loss, params, retain_graph=True, create_graph=False, allow_unused=True)
    grads = [g for g in grads if g is not None] # redundant?
    if len(grads) == 0:
        return torch.tensor(0.0, device=params[0].device)
    
    return torch.sqrt(sum((g ** 2).sum() for g in grads))

optimizer = optim.Adam(model.parameters(), lr=1e-3)

spec = loss_pt.load_specification(
    "specs/mnist-robustness.vcl",
    logic=vcl.VehicleDifferentiableLogic()
)

constraint_loss_fn = spec["robust"]
loss_fn = nn.CrossEntropyLoss()

train_ds = TensorDataset(
    torch.from_numpy(x_train),
    torch.from_numpy(np.argmax(y_train, axis=1)).long(),  # CE wants class indices
)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(NUM_EPOCHS):
    cumulative_loss = 0
    lam = None
    for step, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)

        constraint_loss = torch.stack(constraint_loss_fn(
            n=BATCH_SIZE, 
            classifier=network,
            epsilon=torch.tensor(0.005),
            trainingImages=images.squeeze(1),
            trainingLabels=labels
        )).mean()

        # we only wish to recompute lambda each epoch (as it is expensive)
        if lam is None:
            params = [p for p in model.parameters() if p.requires_grad]
            task_grad_norm = grad_norm(loss, params)
            constraint_grad_norm = grad_norm(constraint_loss, params)
            lam = (task_grad_norm / (task_grad_norm + constraint_grad_norm + 1e-8)).item()

        total_loss = (1 - lam) * loss + lam * constraint_loss
        print(f"{loss.item():.4f}, {constraint_loss.item():.4f}, {total_loss.item():.4f}")

        cumulative_loss += total_loss.item()
        total_loss.backward()
        optimizer.step()

    avg_loss = cumulative_loss / (SUBSET_SIZE / BATCH_SIZE)
    print(
        f"Epoch: {epoch + 1}, "
        f"Lambda: {lam:.4f}, "
        f"Average loss: {avg_loss:.4f}"
    )

classifier = PyTorchClassifier(
    model=model,
    clip_values=(min_pixel_value, max_pixel_value),
    loss=loss_fn,
    optimizer=optimizer,
    input_shape=(1, 28, 28),
    nb_classes=10,
)

preds = classifier.predict(x_test)
accuracy = np.sum(np.argmax(preds, axis=1) == np.argmax(y_test, axis=1)) / len(y_test) 
print(f"Benign accuracy: {accuracy * 100:.1f}%")

tester = CsecTester(model=classifier)
adv_accuracy = tester.run(x_test, y_test, 0.1)
print(f"Constraint security: {adv_accuracy * 100:.1f}%")

exporter = OnnxExporter(filename="pdt_1", classifier=classifier)
exporter.export(torch.randn(1, 1, 28, 28))