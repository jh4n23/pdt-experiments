import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import multiprocessing as mp

from base_classifier import BaseClassifier
from robust_classifier import RobustClassifier
from pdt_classifier import PdtClassifier
from robust_pdt_classifier import RobustPdtClassifier
from torch.utils.data import DataLoader, Subset

def worker(cls_class, train_subset, test_data, num_epochs, batch_size, num_workers, result_queue):
    # Cap threads per process to avoid CPU oversubscription
    torch.set_num_threads(max(1, os.cpu_count() // num_workers))
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

    cls = cls_class()
    name = cls_class.__name__
    try:
        cls.train(train_loader=train_loader, num_epochs=num_epochs, batch_size=batch_size)
        cls.evaluate(test_data=test_data)
        result_queue.put((name, " OK"))
    except Exception as e:
        result_queue.put((name, f" FAILED: {e}"))

def main():
    BATCH_SIZE, SUBSET_SIZE = 64, 1024
    NUM_EPOCHS = 5
    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081))
    ])

    train_data = torchvision.datasets.MNIST(root="./data", train=True, download=False, transform=mnist_transform)
    train_subset = Subset(train_data, range(SUBSET_SIZE))

    test_data = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=mnist_transform)
        
    classifier_classes = [BaseClassifier, RobustClassifier, PdtClassifier, RobustPdtClassifier]
    num_workers = len(classifier_classes)

    result_queue = mp.Queue()
    processes = []

    for cls_class in classifier_classes:
        p = mp.Process(
            target=worker,
            args=(cls_class, train_subset, test_data, NUM_EPOCHS, BATCH_SIZE, num_workers, result_queue),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    while not result_queue.empty():
        name, status = result_queue.get()
        print(f"{name}: {status}")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()