import torch
import warnings
from FRDataset import *
from model import *
from Device import *
from train import *
import torchvision.transforms as T
from torch.utils.data import random_split
from torch.utils.data import DataLoader

torch.manual_seed(43)
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    rename_files('/home/namd/Flower-Recognition/flowers')
    img_size = 64
    stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    transform = T.Compose([T.Resize((img_size, img_size)),
                           T.RandomCrop(64, padding=4, padding_mode='reflect'),
                           T.RandomHorizontalFlip(),
                           T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                           T.ToTensor(),
                           T.Normalize(*stats, inplace=True)])
    dataset = FRDataset('/home/namd/Flower-Recognition/flowers', transform=transform)
    # print(len(dataset)) 4317
    val_pct = 0.1
    val_size = int(val_pct * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, valid_ds = random_split(dataset, [train_size, val_size])

    batch_size = 64
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)
    valid_dl = DataLoader(valid_ds, batch_size * 2, num_workers=3, pin_memory=True)

    device = get_default_device()
    train_dl = DeviceDataLoader(train_dl, device)
    valid_dl = DeviceDataLoader(valid_dl, device)

    # train model
    model = to_device(ResNet50(3, 5), device)
    history = [evaluate(model, valid_dl)]
    epochs = 10
    max_lr = 0.01
    grad_clip = 0.1
    weight_decay = 1e-4
    opt_func = torch.optim.Adam
    history += fit_one_cycle(epochs, max_lr, model, train_dl, valid_dl,
                             grad_clip=grad_clip,
                             weight_decay=weight_decay,
                             opt_func=opt_func)

