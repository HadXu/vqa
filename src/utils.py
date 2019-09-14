import pandas as pd
import json
from keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import torch
from tqdm import tqdm


def load_video_attr(topk=100):
    from collections import Counter
    with open('../input/working/video_attr.json', 'r') as f:
        video_attr = json.loads(f.read())

    v_attr = {}
    num_attr = []
    attr_text = []
    for k, frames_attr in video_attr.items():
        attr_count = [attr for attrs in frames_attr for attr in attrs]
        attr_count = sorted(Counter(attr_count).items(), key=lambda x: x[1], reverse=True)
        attrs = [a for a, freq in attr_count[:topk]]
        num_attr.append(len(attr_count))
        v_attr[k] = attrs
        attr_text += attrs

    attr_text = list(set(attr_text))
    attr2i = {a: i for i, a in enumerate(attr_text)}
    return v_attr, attr_text, attr2i


def question2seq():
    with open('../input/working/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    with open('../input/working/i2q.json', 'r') as f:
        i2q = json.loads(f.read())
    with open('../input/working/q2i.json', 'r') as f:
        q2i = json.loads(f.read())

    word2i = tokenizer.word_index
    video_attr, attr_text, attr2i = load_video_attr(96)

    texts_list = [i2q, attr_text]

    seq_len = (14, 2)

    seq_list = []
    for i, t in enumerate(texts_list):
        q_seq = tokenizer.texts_to_sequences(t)
        q_seq = pad_sequences(q_seq, maxlen=seq_len[i], truncating='post')
        seq_list.append(np.array(q_seq))

    return seq_list, [video_attr, attr2i]


def get_train():
    tr = pd.read_csv('../input/working/train.csv')

    with open('../input/working/i2q.json', 'r') as f:
        i2q = json.loads(f.read())

    with open('../input/working/q2i.json', 'r') as f:
        q2i = json.loads(f.read())

    with open('../input/working/q2c.json', 'r') as f:
        q2c = json.loads(f.read())

    with open('../input/working/i2ans.json', 'r') as f:
        i2ans = json.loads(f.read())

    with open('../input/working/c2v.pkl', 'rb') as f:
        c2v = pickle.load(f)

    y = np.load('../input/working/label.npy')

    [questions, attr_seq], [video_attr, attr2i] = question2seq()
    print(questions.shape, attr_seq.shape)
    print(y.shape)

    train = {}

    data_distr = {'num_q': 0}

    for i, samples in tr.iterrows():
        ques = []
        ans = []
        q_str = []

        prior = []
        for q_i in range(5):
            q = samples['q' + str(q_i)]

            q_seq = questions[q2i[q]] if q != '_UNKQ_' else q

            ques.append(q_seq)
            a_ = [samples['q' + str(q_i) + '_ans' + str(j)] for j in range(3)]
            a_ = [a for a in a_ if a != '_UNKA_']
            ans.append(a_)
            q_str.append(q)

            if q != '_UNKQ_':
                c = q2c[q]
                data_distr[c] = data_distr.get(c, 0) + 1
                data_distr['num_q'] += 1
                prior.append(c2v[c])
            else:
                prior.append(np.zeros(len(i2ans)))

        need_q = [i for i, q in enumerate(ques) if q != '_UNKQ_']

        for q_i in range(5):
            q = samples['q' + str(q_i)]
            if q == '_UNKQ_':
                j = np.random.choice(need_q)
                ques[q_i] = ques[j]
                ans[q_i] = ans[j]
                q_str[q_i] = q_str[j]
                prior[q_i] = prior[j]
                y[i][q_i] = y[i][j]

        v_id = samples['video_id']
        attr = [attr_seq[attr2i[a]] for a in video_attr[v_id]]

        if len(attr) < 96:
            attr += (96 - len(attr)) * [np.zeros(2)]

        assert len(ques) == len(prior) == len(y[i]) == len(q_str) == len(ans) == 5

        train[v_id] = [ques, prior, attr, y[i], q_str, ans]

    return train


def do_train(model, loader, optimizer, fn_loss, device):
    model.train()
    for video, attr, ques, prior, y, _ in tqdm(loader):
        video, attr, ques, prior, y = map(lambda x: x.to(device), [video, attr, ques, prior, y])

        pred = model(video, ques, attr, prior)
        loss = fn_loss(y, pred)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def get_score(ans, index):
    with open('../input/working/i2ans.json', 'r') as f:
        i2ans = json.loads(f.read())

    res = []
    for item in ans:
        r = zip(*item)
        res.append(list(r))

    res = list(zip(*res))

    score = 0
    c = 0
    for ans, idx in zip(res, index):
        for a, i in zip(ans, idx):
            pred_a = i2ans[i]

            if pred_a in a:
                score += 1
            c += 1

    return score / float(c)


def do_valid(model, loader, fn_loss, device):
    model.eval()

    anses = []
    indexs = []
    for video, attr, ques, prior, y, ans in tqdm(loader):
        with torch.no_grad():
            video, attr, ques, prior, y = map(lambda x: x.to(device), [video, attr, ques, prior, y])
            pred = model(video, ques, attr, prior)
            # loss = fn_loss(y, pred)

            index = torch.argmax(pred, dim=2)

            anses += ans
            indexs += index.cpu().numpy().tolist()

    score = get_score(anses, indexs)
    print('score:', score)


if __name__ == '__main__':
    pass
