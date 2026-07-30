import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import AutoProjectedGradientDescent, ProjectedGradientDescent
from art.utils import load_mnist

(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()
x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 10)
)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
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

nb_epochs = 5
batch_size = 64
nb_train = len(x_train)

for epoch in range(nb_epochs):
    perm = np.random.permutation(nb_train)
    x_train_shuffled = x_train[perm]
    y_train_shuffled = y_train[perm]

    for start in range(0, nb_train, batch_size):
        end = start + batch_size
        x_batch = x_train_shuffled[start:end]
        y_batch = y_train_shuffled[start:end]

        x_batch_adv = train_attack.generate(x=x_batch)

        # Combine benign + adversarial versions of this batch
        x_batch_combined = np.concatenate([x_batch, x_batch_adv], axis=0)
        y_batch_combined = np.concatenate([y_batch, y_batch], axis=0)

        # Single gradient step on the combined batch
        classifier.fit(
            x_batch_combined,
            y_batch_combined,
            batch_size=len(x_batch_combined),
            nb_epochs=1,
        )

    preds = classifier.predict(x_test)
    acc = np.sum(np.argmax(preds, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    print(f"Epoch {epoch + 1}/{nb_epochs} - benign test acc: {acc:.4f}")

# --- Final evaluation ---
preds = classifier.predict(x_test)
accuracy = np.sum(np.argmax(preds, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print(f"Final accuracy on benign test data: {accuracy:.4f}")

attack_eval = AutoProjectedGradientDescent(estimator=classifier, eps=0.1)
x_adv = attack_eval.generate(x=x_test)
adv_preds = classifier.predict(x_adv)
adv_accuracy = np.sum(np.argmax(adv_preds, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print(f"Final accuracy on adversarial test data: {adv_accuracy:.4f}")