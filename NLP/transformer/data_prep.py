import random
import torch
import pandas as pd


DATA_PATH = './data/대화체.xlsx'
BATCH_SIZE = 512


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data.loc[idx, '원문'], self.data.loc[idx, '번역문']


def set_seed(seed=12):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)


def data_prep(data_path, batch_size, train_len=85000, valid_len=3000, test_len=2000):
    set_seed()
    data = pd.read_excel(data_path)
    dataset = CustomDataset(data)
    total_len = train_len + valid_len + test_len
    if  total_len == len(dataset):
        train_ds, val_ds, test_ds = torch.utils.data.random_split(dataset, [train_len, valid_len, test_len])
    else:
        train_ds, val_ds, test_ds, _ = torch.utils.data.random_split(dataset, [train_len, valid_len, test_len,
                                                                               len(dataset) - total_len])

    train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=True)
    return train_dataloader, val_dataloader, test_dataloader


if __name__ == '__main__':
    trd, vd, ted = data_prep(DATA_PATH, BATCH_SIZE)
    print(len(trd.dataset))
