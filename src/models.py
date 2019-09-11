import torch
from torch import nn
from torch.utils.data import DataLoader
from loader import VQADataset


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        x_reshape = x.contiguous().view(-1, x.size(-1))

        y = self.module(x_reshape)

        print(y.size())

        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))
        else:
            y = y.view(-1, x.size(1), y.size(-1))

        return y


class VQANet(nn.Module):
    def __init__(self):
        super(VQANet, self).__init__()

        self.emb = nn.Embedding(2243, 300)

        self.video_dp1 = nn.Sequential(
            nn.Dropout2d(0.05),
        )

        self.q_pro = nn.Sequential(
            nn.Dropout(0.05)
        )

        self.time_distribute1 = nn.Sequential(
            TimeDistributed(self.emb),
        )

    def forward(self, video, ques, attr):
        print(video.size())
        print(ques.size())
        print(attr.size())

        print('============')

        q1, q2, q3, q4, q5 = torch.split(ques, 1, dim=1)
        q1, q2, q3, q4, q5 = map(lambda x: torch.squeeze(x, dim=1), [q1, q2, q3, q4, q5])
        q1, q2, q3, q4, q5 = map(self.emb, [q1, q2, q3, q4, q5])
        q1, q2, q3, q4, q5 = map(self.q_pro, [q1, q2, q3, q4, q5])

        v = self.video_dp1(video)
        attr = self.emb(attr)

        print(q1.size())
        print(v.size())
        print(attr.size())

        return v


if __name__ == '__main__':
    names = ['ZJL10000']
    loader = DataLoader(VQADataset(names))

    net = VQANet()

    for video, attr, ques, prior, y, ans in loader:
        pred = net(video, ques, attr)
