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

# get number of allocated CPUs if on HPC cluster,
# or just number of logical CPUs if on local machine
def get_available_cpus():
    num_cpus = None
    if "SLURM_CPUS_PER_TASK" in os.environ:
        num_cpus = int(os.environ["SLURM_CPUS_PER_TASK"])
    try:
        num_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        num_cpus = os.cpu_count()
    finally:
        return num_cpus

def worker(cls_class, train_subset, test_data, num_epochs, batch_size, num_workers, result_queue):
    # set num threads per worker to avoid CPU oversubscription
    torch.set_num_threads(max(1, get_available_cpus() // num_workers))
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
    BATCH_SIZE, SUBSET_SIZE = 64, 4096
    NUM_EPOCHS = 10
    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081))
    ])

    # set download=True if running for the first time
    train_data = torchvision.datasets.MNIST(root="./data", train=True, download=False, transform=mnist_transform)
    train_subset = Subset(train_data, range(SUBSET_SIZE))

    test_data = torchvision.datasets.MNIST(root="./data", train=False, download=False, transform=mnist_transform)
        
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