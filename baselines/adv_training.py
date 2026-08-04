import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import AutoProjectedGradientDescent, ProjectedGradientDescent
from art.utils import load_mnist
from utils.cnn import CNN
from utils.csec_tester import CsecTester

(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()
x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)

model = CNN(in_channels=1, input_size=28, num_classes=10)

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
    print(f"\tEpoch {epoch + 1}/{nb_epochs} - benign test acc: {acc:.4f}")

# Final evaluation
print("Final evaluation")
preds = classifier.predict(x_test)
accuracy = np.sum(np.argmax(preds, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print(f"\tBenign accuracy: {accuracy:.4f}")

tester = CsecTester(model=classifier)
adv_accuracy = tester.run(x_test, y_test, 0.1)
print(f"\tAdversarial accuracy: {adv_accuracy:.4f}")