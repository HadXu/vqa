import torch
from torch import nn
from torch.utils.data import DataLoader
from loader import VQADataset
from torch.nn import functional as F


def distance(q, v, dist, dim=2):
    if dist == 'dice':
        return q * v / (torch.sum(q ** 2, dim=dim, keepdim=True) + torch.sum(v ** 2, dim=dim, keepdim=True))


def temporal_attention(q, v, use_conv=False):
    def get_atten_w(q_list, v_encode, fca):
        w_list = []
        for q in q_list:
            merged = distance(q, v_encode, 'dice')

            w = fca(merged)
            w = w.view(-1)
            w = F.softmax(w)
            w_list.append(w)

        w = torch.mean(torch.cat(w_list))
        w = w.view(-1, 1, 1)
        return w

    q = map(lambda x: torch.mean(x, dim=1), q)

    q = map(nn.Linear(300, 256), q)
    q = map(lambda x: torch.reshape(x, (-1, 1, 256)), q)

    fca = nn.Linear(256, 1)

    v_encode = torch.mean(v, dim=2)
    v_encode = nn.Linear(2048, 256)(v_encode)

    w = get_atten_w(q, v_encode, fca)

    return v * w


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

    def forward(self, video, ques, attr, prior):
        print(video.size())
        print(ques.size())
        print(attr.size())
        print(prior.size())

        print('============')

        q1, q2, q3, q4, q5 = torch.split(ques, 1, dim=1)
        q1, q2, q3, q4, q5 = map(lambda x: torch.squeeze(x, dim=1), [q1, q2, q3, q4, q5])
        q1, q2, q3, q4, q5 = map(self.emb, [q1, q2, q3, q4, q5])
        q1, q2, q3, q4, q5 = map(self.q_pro, [q1, q2, q3, q4, q5])

        v = self.video_dp1(video)
        attr = self.emb(attr)

        v = temporal_attention([q1, q2, q3, q4, q5], v)

        print(v.size())
        print(attr.size())

        return v


if __name__ == '__main__':
    names = ['ZJL10000']
    loader = DataLoader(VQADataset(names))

    net = VQANet()

    for video, attr, ques, prior, y, ans in loader:
        pred = net(video, ques, attr, prior)
