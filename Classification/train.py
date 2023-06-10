import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
import time

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_model(model, train_dataloader, valid_dataloader, criterion, optimizer, epochs,
                save_model_path, save_history_path, batch_size, train_ratio, verbose=True,
                **kwargs):

    if "LR_STEP" in kwargs:
        scheduler = StepLR(optimizer, step_size=kwargs['LR_STEP'], gamma=kwargs['LR_GAMMA'])
    else:
        scheduler = None

    history = {'train_loss': [], 'train_acc': [], 'valid_loss': [], 'valid_acc': []}
    best_loss = 999999999
    for ep in range(epochs):
        epoch_start = time.time()

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch: {ep+1}, current_LR = {current_lr}")

        model.train()    # train mode
        train_loss, train_acc, _ = step_epoch(model, train_dataloader, criterion, optimizer)

        model.eval()     # test mode
        with torch.no_grad():
            valid_loss, valid_acc, _ = step_epoch(model, valid_dataloader, criterion)

        history['train_loss'] += [train_loss]
        history['valid_loss'] += [valid_loss]
        history['train_acc'] += [train_acc]
        history['valid_acc'] += [valid_acc]

        if verbose:
            if (ep + 1) % verbose == 0:
                print('-' * 30)
                print(f"Epoch: {ep + 1} Done! - time: {time.time() - epoch_start:.2f}s\n"
                      f"loss - train: {train_loss:.3f} / valid: {valid_loss:.3f}"
                      f" & accuaracy - train: {train_acc:.3f} / valid: {valid_acc:.3f}")

        if valid_loss < best_loss:
            print('=== Best Loss - Save this model ===')
            best_loss = valid_loss
            torch.save({"model": model.state_dict(),
                        "epoch": ep,
                        "optimizer": optimizer,
                        "scheduler": scheduler,
                        },
                       save_model_path)

        if 'LR_STEP' in kwargs:
            scheduler.step()

    torch.save({'history': history,
                'epoch': epochs,
                'batch_size': batch_size,
                'train_ratio': train_ratio}, save_history_path)

    return history


def step_epoch(model, dataloader, criterion, optimizer=None):
    running_loss = 0
    correct = 0
    total_n = len(dataloader.dataset)

    for data in tqdm(dataloader, leave=False):
        x_batch, y_batch = data
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        if model.__class__.__name__ == 'InceptionNet' and model.training:
            y_hat, out1, out2 = model(x_batch)
            network_loss = criterion(y_hat, y_batch)
            aux1_loss = criterion(out1, y_batch)
            aux2_loss = criterion(out2, y_batch)
            loss = network_loss + 0.3*aux1_loss + 0.3*aux2_loss
        else:
            # inference
            y_hat = model(x_batch)
            # loss
            loss = criterion(y_hat, y_batch)

        if optimizer is not None:
            # update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # loss accumulation
        loss_batch = loss.item() * x_batch.size(0)
        running_loss += loss_batch

        # for accuracy
        y_hat = y_hat.argmax(dim=1)
        correct += torch.sum(y_hat == y_batch).item()

    epoch_loss = running_loss / total_n
    epoch_accuracy = correct / total_n
    return epoch_loss, epoch_accuracy, correct


def test_model(model, test_dataloader, criterion):
    model.eval()
    n = len(test_dataloader.dataset)
    loss, acc, correct = step_epoch(model, test_dataloader, criterion)
    print("Test - Loss: {:.3f} // Accuracy: {:,}/{:,} ({:.2f}%)".format(loss, correct, n, 100*acc))
    return round(acc, 1)


if __name__ == '__main__':
    y_hat = torch.tensor([0, 1, 1, 0])
    y_batch = torch.tensor([0, 1, 1, 1])

    print(torch.sum(y_hat == y_batch).item())
    print((y_hat == y_batch).sum())

