import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

TRAIN_RATIO = 0.800
BATCH_SIZE = 512
DATASET = 'CIFAR10'


def load_dataset(dataset, train_transform, test_transform, batch_size, train_ratio=None, root='./data'):
    if dataset == 'CIFAR10':
        dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=train_transform)
        test_data = datasets.CIFAR10(root=root, train=False, download=True, transform=test_transform)
    elif dataset == 'MNIST':
        dataset = datasets.MNIST(root=root, train=True, download=True, transform=train_transform)
        test_data = datasets.MNIST(root=root, train=False, download=True, transform=test_transform)
    elif dataset == 'STL10':
        dataset = datasets.STL10(root=root, split='train', download=True, transform=train_transform)
        test_data = datasets.STL10(root=root, split='test', download=True, transform=test_transform)
    else:
        raise ValueError("Invaild DATASET NAME!")

    if train_ratio is None:
        train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        return train_dataloader, test_dataloader
    else:
        # random split - train / validation
        train_data, valid_data = torch.utils.data.random_split(dataset, [round(train_ratio, 1), round(1 - train_ratio, 1)])
        valid_data.transform = test_transform

        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        return train_dataloader, valid_dataloader, test_dataloader


if __name__ == '__main__':
    train_transform = transforms.ToTensor()
    test_transform = transforms.ToTensor()
    train_dl, valid_dl, test_dl = load_dataset('CIFAR10', train_transform=train_transform, test_transform=test_transform,
                                     batch_size=32, train_ratio=0.8)

    x_train, y_train = next(iter(train_dl))
    x_valid, y_valid = next(iter(valid_dl))
    x_test, y_test = next(iter(test_dl))
    print(x_train.shape, y_train.shape)
    print(x_valid.shape, y_valid.shape)
    print(x_test.shape, y_test.shape)