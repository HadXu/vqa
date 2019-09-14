import torch
from torch import nn
from torch.nn import functional as F


def distance(q, v, dist, dim=2):
    if dist == 'dice':
        return q * v / (torch.sum(q ** 2, dim=dim, keepdim=True) + torch.sum(v ** 2, dim=dim, keepdim=True))


def temporal_attention(q, v):
    def get_atten_w(q_list, v_encode, fca):
        w_list = []
        for q in q_list:
            merged = distance(q, v_encode, 'dice')

            w = fca(merged)
            w = w.view(-1)
            w = F.softmax(w, dim=0)
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

        self.q_process = nn.Sequential(
            self.emb,
            nn.Dropout(0.05)
        )

        self.video_fc = nn.Linear(2048, 300)
        self.fc = nn.Linear(1200, 951)

    def forward(self, video, ques, attr, prior):
        q1, q2, q3, q4, q5 = torch.split(ques, 1, dim=1)
        q1, q2, q3, q4, q5 = map(lambda x: torch.squeeze(x, dim=1), [q1, q2, q3, q4, q5])
        q1, q2, q3, q4, q5 = map(self.q_process, [q1, q2, q3, q4, q5])

        v = self.video_dp1(video)
        attr = self.emb(attr)

        v = temporal_attention([q1, q2, q3, q4, q5], v)

        v = torch.mean(v, dim=2)

        v = self.video_fc(v)

        attr = torch.mean(attr, dim=2)
        q1, q2, q3, q4, q5 = map(lambda x: torch.mean(x, dim=1), [q1, q2, q3, q4, q5])  # bs*300

        a1, a2, a3, a4, a5 = map(lambda x: x.unsqueeze(dim=1) * attr, [q1, q2, q3, q4, q5])  # bs*96*300
        a1, a2, a3, a4, a5 = map(lambda x: torch.mean(x, dim=1), [a1, a2, a3, a4, a5])  # bs*300

        q = torch.stack([q1, q2, q3, q4, q5], dim=1)  # bs * 5 * 300
        attr = torch.stack([a1, a2, a3, a4, a5], dim=1)  # bs * 5 * 300

        v1 = torch.mean(q.unsqueeze(dim=2) * v.unsqueeze(dim=1), dim=2)
        v2 = torch.mean(attr.unsqueeze(dim=2) * v.unsqueeze(dim=1), dim=2)

        fc = torch.cat([q, attr, v1, v2], dim=2)

        out = prior * torch.sigmoid(self.fc(fc))

        return out


if __name__ == '__main__':
    video = torch.randn(2, 16, 36, 2048)
    attr = torch.randint(100, size=(2, 96, 2)).long()
    ques = torch.randint(100, size=(2, 5, 14)).long()
    prior = torch.randint(1, size=(2, 5, 951)).float()

    net = VQANet()

    pred = net(video, ques, attr, prior)

    print(pred.size())
