import numpy as np
from torch.utils.data import DataLoader, Dataset
from utils import get_train
import torch


class VQADataset(Dataset):
    def __init__(self, names):
        super(VQADataset, self).__init__()
        self.names = names
        self.train_dict = get_train()

    def __getitem__(self, x):
        name = self.names[x]
        video = np.load(f'../input/rcnn/{name}.npy')

        index = []
        for i in np.linspace(0, 40 - 1, 16):
            j = np.random.randint(int(i), min(40, int(i + 3)))
            index.append(j)

        video = video[np.array(index)]

        ques, prior, attr, y, q_str, ans = self.train_dict[name]

        prior = np.array(prior)

        ques = np.array(ques)
        attr = np.array(attr)

        return torch.from_numpy(video).float(), torch.from_numpy(attr).long(), torch.from_numpy(ques).long(), \
               torch.from_numpy(prior).float(), torch.from_numpy(y).float(), ans

    def __len__(self):
        return len(self.names)


if __name__ == '__main__':
    names = ['ZJL10000', 'ZJL10000']
    loader = DataLoader(VQADataset(names), batch_size=2)
    for video, attr, ques, prior, y, _ in loader:
        print(video.size())
        print(attr.size())
        print(ques.size())
        print(prior.size())
        print(y.size())
