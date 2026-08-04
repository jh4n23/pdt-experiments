import torch
import onnx
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import AutoProjectedGradientDescent, ProjectedGradientDescent
from art.utils import load_mnist
from cnn import CNN
from csec_tester import CsecTester
from onnx_exporter import OnnxExporter

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

classifier.fit(x_train, y_train, batch_size=64, nb_epochs=5)
preds = classifier.predict(x_test)
accuracy = np.sum(np.argmax(preds, axis=1) == np.argmax(y_test, axis=1)) / len(y_test) 
print(f"Benign accuracy: {accuracy * 100:.1f}%")

tester = CsecTester(model=classifier)
adv_accuracy = tester.run(x_test, y_test, 0.1)
print(f"Constraint security: {adv_accuracy * 100:.1f}%")

exporter = OnnxExporter(filename="standard_baseline", classifier=classifier)
exporter.export(torch.randn(1, 1, 28, 28))