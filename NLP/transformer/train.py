import os
import math
import torch


from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer

from model.transfomer import Transformer


def train(model, epochs, tokenizer, max_len, train_dataloader, valid_dataloader, criterion, optimizer, scheduler, device, model_path):
    os.makedirs(model_path, exist_ok=True)
    model_save_path = os.path.join(model_path, 'best_model.pt')
    history_save_path = os.path.join(model_path, 'best_model_history.dict')

    loss_history = defaultdict(list)
    best_loss = 999999999

    for epoch in range(epochs):
        model.train()

        train_loss = step_epoch(model, tokenizer, max_len, train_dataloader, criterion, device, optimizer, scheduler)
        loss_history['train'].append(train_loss)

        model.eval()
        with torch.no_grad():
            val_loss = step_epoch(model, tokenizer, max_len, valid_dataloader, criterion, device)
            loss_history['valid'].append(val_loss)
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({
                    "model": model,
                    "epoch": epoch,
                    "optimizer": optimizer,
                    "scheduler": scheduler
                }, model_save_path)

        print(f"[EPOCH ({epoch+1}/{epochs})] - current_lr: {optimizer.param_groups[0]['lr']:.5f} | "
              f"Loss: {train_loss:.3f}/{val_loss:.3f}")
        print('='*20)

    torch.save({"loss_history": loss_history,
                "EPOCH": epochs
                }, history_save_path)


def test(model, test_dataloader, criterion):
    model.eval()
    with torch.no_grad():
        test_loss = step_epoch(model, test_dataloader, criterion)
    print(f"Test loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):.3f}")


def step_epoch(model, tokenizer, max_len, dataloader, criterion, device, optimizer=None, scheduler=None):
    running_loss = 0.0
    total_n = len(dataloader.dataset)

    for source, target in tqdm(dataloader, leave=False):
        x, y = make_tokens_to_train(source, target, tokenizer, max_len)
        x, y = x.to(device), y.to(device)

        y_pred = model(x, y[:, :-1])  # target의 마지막 토큰 제외

        loss = criterion(y_pred.permute(0, 2, 1), y[:, :-1])  # loss 계산 시, <sos> token 제외

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        batch_loss = loss.item() * x.shape[0]
        running_loss += batch_loss

    epoch_loss = running_loss / total_n
    return epoch_loss


def make_tokens_to_train(source, target, tokenizer, max_len):
    x = tokenizer(source, padding=True, truncation=True, max_length=max_len, return_tensors='pt').input_ids
    y = [tokenizer.eos_token + text for text in target]  # add sos token
    y = tokenizer(y, padding=True, truncation=True, max_length=max_len, return_tensors='pt').input_ids
    return x, y




if __name__ == '__main__':


    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ko-en")

    source = ['돼지같은 놈', '좋은 하루 보내세요', '모자라면 자동으로 패딩을 붙여주나봐']
    target = ['You like a pig', 'Have a good day', 'Yes Maybe add pad token to sentence if the source length is short']

    model = Transformer(tokenizer.vocab_size, 100, tokenizer.pad_token_id, 512, 8, 2048, 0.1, 6, 'cuda')


