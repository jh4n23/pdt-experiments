import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import AutoProjectedGradientDescent
from art.utils import load_mnist
from cnn import CNN
from csec_tester import CsecTester

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

print("Initial training stage")
classifier.fit(x_train, y_train, batch_size=64, nb_epochs=5)
preds = classifier.predict(x_test)
accuracy = np.sum(np.argmax(preds, axis=1) == np.argmax(y_test, axis=1)) / len(y_test) 
print(f"\tBenign accuracy: {accuracy:.4f}")

attack = AutoProjectedGradientDescent(estimator=classifier, eps=0.1)

# Generate adv training data, combine with benign, and train new classifier on both sets of data
print("Second training stage: benign + adversarial test data")
x_train_adv = attack.generate(x=x_train)
 
x_combined = np.concatenate([x_train, x_train_adv], axis=0)
y_combined = np.concatenate([y_train, y_train], axis=0)

perm = np.random.permutation(len(x_combined))
x_combined = x_combined[perm]
y_combined = y_combined[perm]

classifier.optimizer.zero_grad()
classifier.fit(x_combined, y_combined, batch_size=64, nb_epochs=5)

# Evaluate on both benign and adversarial training data
print("Final evaluation")
preds = classifier.predict(x_test)
benign_accuracy = np.sum(np.argmax(preds, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print(f"\tBenign accuracy: {benign_accuracy}")

tester = CsecTester(model=classifier)
adv_accuracy = tester.run(x_test, y_test, 0.1)
print(f"\tAdversarial accuracy: {adv_accuracy:.4f}")