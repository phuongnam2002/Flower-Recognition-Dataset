import torch
import torch
from torch.nn import *
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
from tensorflow.keras import backend


def accuracy(outputs, labels):
    logits, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == lables).item() / len(preds))


class ImageClassificationBase(Module):
    def __init__(self):
        super(ImageClassificationBase, self).__init__()

    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # predic
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracy
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}],{} train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, "last_lr: {:.5f},".format(result['lrs'][-1]) if 'lrs' in result else '',
            result['train_loss'], result['val_loss'], result['val_acc']))


def conv_block(in_channels, out_channels, pooling=0):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if pooling:
        layers.append((nn.MaxPool2d(2)))
    return nn.Sequential(*layers)


class ResNet50(ImageClassificationBase):
    def __init__(self, in_channels, num_classes):
        super(ResNet50, self).__init__()

        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pooling=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        self.conv3 = conv_block(128, 256, pooling=True)
        self.conv4 = conv_block(256, 512, pooling=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out


if __name__ == '__main__':
    pass
