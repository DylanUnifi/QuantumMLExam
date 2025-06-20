# data_loader/mnist.py
from torchvision import datasets, transforms
import torch

def load_mnist(batch_size=64, binary_classes=(3, 5), grayscale=True):
    transform = transforms.Compose([
        transforms.Grayscale() if grayscale else transforms.Lambda(lambda x: x),
        transforms.ToTensor()
    ])

    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    def filter_classes(dataset):
        X, y = [], []
        for img, label in dataset:
            if label in binary_classes:
                X.append(img.view(-1))
                y.append(1 if label == binary_classes[1] else 0)
        return torch.stack(X), torch.tensor(y, dtype=torch.float32)

    X_train, y_train = filter_classes(train_set)
    X_test, y_test = filter_classes(test_set)

    return X_train, y_train, X_test, y_test