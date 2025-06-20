# data_loader/svhn.py
from torchvision import transforms
from torchvision.datasets import SVHN
import torch

def load_svhn(batch_size=64, binary_classes=(3, 5), grayscale=True):
    transform = transforms.Compose([
        transforms.Grayscale() if grayscale else transforms.Lambda(lambda x: x),
        transforms.ToTensor()
    ])

    train_set = SVHN(root='./data', split='train', download=True, transform=transform)
    test_set = SVHN(root='./data', split='test', download=True, transform=transform)

    def filter_classes(dataset):
        X, y = [], []
        for img, label in zip(dataset.data, dataset.labels):
            if label in binary_classes:
                tensor_img = transform(transforms.ToPILImage()(img)).view(-1)
                X.append(tensor_img)
                y.append(1 if label == binary_classes[1] else 0)
        return torch.stack(X), torch.tensor(y, dtype=torch.float32)

    X_train, y_train = filter_classes(train_set)
    X_test, y_test = filter_classes(test_set)

    return X_train, y_train, X_test, y_test