import torch
import torch.optim as optim
import torch.nn as nn

from transformers import AutoTokenizer

from utils import NoamScheduler
from data_prep import data_prep
from train import train
from model.transfomer import Transformer


DATA_PATH = './data/대화체.xlsx'
BATCH_SIZE = 64
TOKENIZER_REPO = "Helsinki-NLP/opus-mt-ko-en"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# tokenizers
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_REPO)
vocab_size = tokenizer.vocab_size
pad_idx = tokenizer.pad_token_id

# model parameters
max_len = 100
d_model = 256
n_heads = 8
d_ff = 1024
drop_p = 0.1
n_layers = 6

# scehduler parameters
warmup_steps = 1000
lr_scale = 0.5

# train parameters
epochs = 20
model_path = './save'

train_dataloader, valid_dataloader, test_dataloader = data_prep(DATA_PATH, BATCH_SIZE, train_len=80000)
model = Transformer(vocab_size, max_len, pad_idx, d_model, n_heads, d_ff, drop_p, n_layers, device).to(device)
optimizer = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
scheduler = NoamScheduler(optimizer, d_model=d_model, warmup_steps=warmup_steps, LR_scale=lr_scale)
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)


def main():
    train(model, epochs, tokenizer, max_len, train_dataloader, valid_dataloader, criterion, optimizer, scheduler,
          device, model_path)


if __name__ == '__main__':
    main()


