import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


from base_classifier import BaseClassifier
from robust_classifier import RobustClassifier
from pdt_classifier import PdtClassifier
from robust_pdt_classifier import RobustPdtClassifier
from utils.cnn import CNN
from torch.utils.data import DataLoader, Subset
from utils.csec_tester import CsecTester

"""
What should this file do?
    - Create a BaseClassifier
    - Create a RobustClassifier
    - Create a PdtClassifier 
    - Create a RobustPdtClassifier

    - Train each classifier on the same data
    - Test each classifier on the same data, reporting accuracy and constraint security
"""
def main():
    BATCH_SIZE, SUBSET_SIZE = 64, 1024
    NUM_EPOCHS = 5
    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081))
    ])

    train_data = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=mnist_transform)
    train_subset = Subset(train_data, range(SUBSET_SIZE))
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)

    test_data = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=mnist_transform)

    baseCls = BaseClassifier()
    robustCls = RobustClassifier()
    pdtCls = PdtClassifier()
    robustPdtCls = RobustPdtClassifier()
        
    classifiers = [baseCls]

    for cls in classifiers:
        cls.train(train_loader=train_loader, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
        cls.evaluate(test_data=test_data)

if __name__ == "__main__":
    main()