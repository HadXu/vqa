import numpy as np


def get_features():
    # fea = np.load('../input/rcnn/ZJL10000.npy')
    # print(fea.shape)
    label = np.load('../input/working/label.npy')
    print(label.shape)


if __name__ == '__main__':
    ans = [[('mirror', 'mirror'), ('people', 'people')],
           [('bathroom', 'bathroom'), ('in front of mirror', 'in front of mirror'), ('indoor', 'indoor')],
           [('yellow', 'yellow'), ('pale yellow', 'pale yellow'), ('light yellow', 'light yellow')],
           [('speaking', 'speaking')],
           [('two', 'two')]]


