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


l = {'ZJL459', 'ZJL3387', 'ZJL1543', 'ZJL268', 'ZJL2303', 'ZJL1523', 'ZJL1559', 'ZJL2536', 'ZJL2597', 'ZJL7931',
     'ZJL2261', 'ZJL1376', 'ZJL2245', 'ZJL2580', 'ZJL1295', 'ZJL2416', 'ZJL2305', 'ZJL2413', 'ZJL1572', 'ZJL8014',
     'ZJL2246', 'ZJL2229', 'ZJL1265', 'ZJL2615', 'ZJL870', 'ZJL2243', 'ZJL2503', 'ZJL1458', 'ZJL4292', 'ZJL787',
     'ZJL1563', 'ZJL293', 'ZJL2560', 'ZJL4193', 'ZJL5122', 'ZJL793', 'ZJL1344', 'ZJL2232', 'ZJL7988', 'ZJL206',
     'ZJL1509', 'ZJL1493', 'ZJL2411', 'ZJL2473', 'ZJL2476', 'ZJL2547', 'ZJL2646', 'ZJL2279', 'ZJL3361', 'ZJL3430',
     'ZJL2240', 'ZJL2242', 'ZJL831', 'ZJL1512', 'ZJL1610', 'ZJL1280', 'ZJL2550', 'ZJL7956', 'ZJL1281', 'ZJL2529',
     'ZJL2542', 'ZJL2543', 'ZJL1547', 'ZJL1557', 'ZJL2584', 'ZJL2590', 'ZJL1593', 'ZJL2552', 'ZJL2601', 'ZJL2816',
     'ZJL2221', 'ZJL1780', 'ZJL2486', 'ZJL886', 'ZJL2248', 'ZJL1435', 'ZJL1578', 'ZJL8298', 'ZJL883', 'ZJL2665',
     'ZJL1592', 'ZJL2236', 'ZJL2254', 'ZJL2295', 'ZJL2569', 'ZJL7986', 'ZJL1143', 'ZJL1307', 'ZJL1609', 'ZJL2696',
     'ZJL2649', 'ZJL1145', 'ZJL2414', 'ZJL2225'}


def main():
    print('~~~~~~~~~~ start training ~~~~~~~~~~~')
    # names = ['ZJL10000', 'ZJL10000']

    names = os.listdir('../input/rcnn/')
    names = [x.split('.')[0] for x in names]
    names = [x for x in names if x not in l]

    tr_names = names[:8000]
    val_names = names[8000:]

    tr_loader = DataLoader(VQADataset(tr_names), batch_size=32)
    val_loader = DataLoader(VQADataset(val_names), batch_size=32)

    model = VQANet().to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)

    for e in range(100):
        do_train(model, tr_loader, optimizer, focal_loss_fixed, device=device)
        do_valid(model, val_loader, focal_loss_fixed, device=device)


if __name__ == '__main__':
    main()
