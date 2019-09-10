import numpy as np
from torch.utils.data import DataLoader, Dataset
from utils import get_train
import json


class VQADataset(Dataset):
    def __init__(self, names):
        super(VQADataset, self).__init__()
        self.names = names
        self.train_dict = get_train()

    def __getitem__(self, x):
        name = self.names[x]
        video = np.load(f'../input/rcnn/{name}.npy')

        ques, prior, attr, y = self.train_dict[name]

        prior = np.array(prior)
        ques = np.array(ques)
        attr = np.array(attr)

        return video, attr, ques, prior, y

    def __len__(self):
        return len(self.names)


if __name__ == '__main__':
    names = ['ZJL10000']
    loader = DataLoader(VQADataset(names))
    for video, attr, ques, prior, y in loader:
        print(video.size())
        print(attr.size())
        print(ques.size())
        print(prior.size())
        print(y.size())
