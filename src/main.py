from torch import nn
import torch
from torch.utils.data import DataLoader
from loader import VQADataset
from utils import do_train, do_valid
from models import VQANet
from torch.optim import Adam
import os

device = torch.device('cpu')


def focal_loss_fixed(y_true, y_pred):
    alpha = 0.65

    z = torch.sum(y_true) + torch.sum(1 - y_true)

    pt_1 = torch.where(torch.ge(y_true, 0.5), y_pred, torch.ones_like(y_pred))
    pt_0 = torch.where(torch.le(y_true, 0.5), y_pred, torch.zeros_like(y_pred))

    pos_loss = -torch.sum(alpha * torch.pow(1. - pt_1, 2) * torch.log(pt_1))
    neg_loss = -torch.sum((1 - alpha) * torch.pow(pt_0, 2) * torch.log(1. - pt_0))

    return (pos_loss + neg_loss) / z


def main():
    print('~~~~~~~~~~ start training ~~~~~~~~~~~')
    # names = ['ZJL10000', 'ZJL10000']

    names = os.listdir('../input/rcnn/')
    names = [x.split('.')[0] for x in names]

    tr_names = names[:8000]
    val_names = names[8000:]

    tr_loader = DataLoader(VQADataset(tr_names), batch_size=2)
    val_loader = DataLoader(VQADataset(val_names), batch_size=2)

    model = VQANet().to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)

    for e in range(100):
        do_train(model, tr_loader, optimizer, focal_loss_fixed, device=device)
        do_valid(model, val_loader, focal_loss_fixed, device=device)


if __name__ == '__main__':
    main()
