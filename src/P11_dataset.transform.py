import torchvision

train_set = torchvision.datasets.CIFAR10(root="./dataset",train=True,download=True)