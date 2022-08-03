import torch
import tqdm
from FRDataset import *
import torch.nn as nn
import torch.nn.functional as F


def save_model(model, path, name):
    torch.save(model.state_dict(), os.path.join(path, name))


def evaluate(model, val_loader):
    model.eval()
    out = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(out)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.Adam):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # training phase
        model.train()
        train_loss = []
        for batch in tqdm(train_loader):
            loss = model.training_step(batch)
            train_loss.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # validation phase
            result = evaluate(model, val_loader)
            result['train_loss'] = torch.stack(train_loss).mean().item()
            model.epoch_end(epoch, result)
            history.append(result)

    return history


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader,
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.Adam):
    torch.cuda.empty_cache()
    history = []

    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps=train_loader)

    for epoch in range(epochs):
        model.train()
        train_losses = []
        lrs = []
        for batch in tqdm(train_loader):
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()

            # Gradient clipping
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()

            lrs.append(get_lr(optimizer))
            sched.step()

        # validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
    return history


if __name__ == '__main__':
    pass
