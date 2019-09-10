import numpy as np


def get_features():
    # fea = np.load('../input/rcnn/ZJL10000.npy')
    # print(fea.shape)
    label = np.load('../input/working/label.npy')
    print(label.shape)


if __name__ == '__main__':
    get_features()
