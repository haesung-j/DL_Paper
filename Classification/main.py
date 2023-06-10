from ast import Num

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from load_dataset import load_dataset
from train import train_model, test_model
from models.VGGnet import VGGnet
from models.InceptionNet import InceptionNet

# data
DATASET = 'CIFAR10'
TRAIN_RATIO = 0.8
BATCH_SIZE = 1024
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# model
MODEL_NAME = 'InceptionNet'
config = 'D'
EPOCHS = 50
LR = 1e-3
BATCH_NORMALIZATION = True
NUM_CLASSES = 10
INIT_WEIGHTS = True
DROPOUT = 0.5
SAVE_MODEL_PATH = f'./results/{DATASET}_{MODEL_NAME}.pt'
SAVE_HISTORY_PATH = f'./results/{DATASET}_{MODEL_NAME}_HISTORY.dict'

# Only test
MODEL_TRAINING = False

# Load data
train_transform = transforms.ToTensor()
test_transform = transforms.ToTensor()
train_dl, valid_dl, test_dl = load_dataset(DATASET, train_transform=train_transform, test_transform=test_transform,
                                           train_ratio=TRAIN_RATIO, batch_size=BATCH_SIZE)

# Load model
# configurations = {'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#                   'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#                   'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#                   'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
#                  }
# model = VGGnet(configurations[config], BATCH_NORMALIZATION, NUM_CLASSES, INIT_WEIGHTS, DROPOUT).to(DEVICE)
model = InceptionNet().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

if MODEL_TRAINING:
    # Training
    train_model(model, train_dl, valid_dl, criterion, optimizer, EPOCHS,
                SAVE_MODEL_PATH, SAVE_HISTORY_PATH, BATCH_SIZE, TRAIN_RATIO)
else:
    model.load_state_dict(torch.load(SAVE_MODEL_PATH, map_location=DEVICE)['model'])
    test_model(model, test_dl, criterion)
    quit()

# test
# best_model = VGGnet(configurations[config], BATCH_NORMALIZATION, NUM_CLASSES, INIT_WEIGHTS, DROPOUT).to(DEVICE)
best_model = InceptionNet().to(DEVICE)
test_model(best_model, test_dl, criterion)

