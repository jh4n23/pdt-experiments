import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import vehicle_lang as vcl

from tqdm import tqdm
from vehicle_lang.loss import pytorch as loss_pt
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import AutoProjectedGradientDescent, ProjectedGradientDescent
from art.utils import load_mnist
from utils.cnn import CNN
from utils.csec_tester import CsecTester

BATCH_SIZE = 64
SUBSET_SIZE = 1024
NUM_EPOCHS = 5

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

classifier = PyTorchClassifier(
    model=model,
    clip_values=(min_pixel_value, max_pixel_value),
    loss=loss_fn,
    optimizer=optimizer,
    input_shape=(1, 28, 28),
    nb_classes=10,
)

train_attack = ProjectedGradientDescent(
    estimator=classifier,
    eps=0.1,
    eps_step=0.01,
    max_iter=10,
)

for epoch in range(NUM_EPOCHS):
    perm = np.random.permutation(SUBSET_SIZE)
    x_train_shuffled = x_train[perm]
    y_train_shuffled = y_train[perm]
    cumulative_loss = 0
    lam = None

    pbar = tqdm(range(0, SUBSET_SIZE, BATCH_SIZE), desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for start in pbar:
        end = start + BATCH_SIZE
        x_batch = x_train_shuffled[start:end]
        y_batch = y_train_shuffled[start:end]

        x_batch_adv = train_attack.generate(x=x_batch)

        # Combine benign + adversarial versions of this batch
        x_batch_combined = np.concatenate([x_batch, x_batch_adv], axis=0)
        y_batch_combined = np.concatenate([y_batch, y_batch], axis=0)

        images = torch.from_numpy(x_batch_combined)
        labels = torch.from_numpy(np.argmax(y_batch_combined, axis=1)).long()

        optimizer.zero_grad()
        logits = model(images)
        task_loss = loss_fn(logits, labels)

        constraint_loss = torch.stack(constraint_loss_fn(
            n=images.shape[0], 
            classifier=network,
            epsilon=torch.tensor(0.005),
            trainingImages=images.squeeze(1),
            trainingLabels=labels
        )).mean()

        if lam is None:
            params = [p for p in model.parameters() if p.requires_grad]
            task_grad_norm = grad_norm(task_loss, params)
            constraint_grad_norm = grad_norm(constraint_loss, params)
            lam = (task_grad_norm / (task_grad_norm + constraint_grad_norm + 1e-8)).item()

        total_loss = (1 - lam) * task_loss + lam * constraint_loss

        cumulative_loss += total_loss.item()
        total_loss.backward()
        optimizer.step()
        pbar.set_postfix(task_loss=task_loss.item(), constraint_loss=constraint_loss.item(), total_loss=total_loss.item(), lam=lam)

    avg_loss = cumulative_loss / (SUBSET_SIZE / BATCH_SIZE)
    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS}: "
        f"\tLambda: {lam:.4f}, "
        f"\tAverage total loss: {avg_loss:.4f}"
    )

    preds = classifier.predict(x_test)
    acc = np.sum(np.argmax(preds, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    print(f"\tEpoch {epoch + 1}/{NUM_EPOCHS} - benign test acc: {acc:.4f}")

# Final evaluation
print("Final evaluation")
preds = classifier.predict(x_test)
accuracy = np.sum(np.argmax(preds, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print(f"\tBenign accuracy: {accuracy:.4f}")

tester = CsecTester(model=classifier)
adv_accuracy = tester.run(x_test, y_test, 0.1)
print(f"\tAdversarial accuracy: {adv_accuracy:.4f}")