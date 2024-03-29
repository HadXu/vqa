import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
tf_cfg = tf.ConfigProto(allow_soft_placement=True)
tf_cfg.gpu_options.allow_growth=True
session = tf.Session(config=tf_cfg)
KTF.set_session(session)
from utils import *
from model import *
from keras import backend as K
from sklearn.cross_validation import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')
from config import *



def main(cfg):
    def select_bestepoch(result):
        submit = 0

        for answers in result:
            ans_ = sorted(answers, key=lambda x: x[1])
            pred = (ans_[-3][0] + ans_[-2][0] + ans_[-1][0]) / 3
            submit += pred
        submit /= len(result)
        return submit

    bs = cfg['bs']
    patience = cfg['patience']
    cfg['frame_size'] = 2048
    tr, te, y, embed_matrix, cfg = load_data('glove42',cfg)


    num_fold = 5
    folds = StratifiedKFold(select_label(y,True), num_fold, shuffle=True, random_state=cfg['seed'])
    te_g = DataLoader(False, te,None,cfg)
    print(cfg)
    for n_fold, (tr_idx, val_idx) in enumerate(folds):
        #if n_fold not in cfg['fold']:
        #    continue

        print(f'============fold {n_fold}==============')
        tr_x = [tr[i] for i in tr_idx]
        tr_y = y[tr_idx]
        tr_g = DataLoader(True, tr_x, tr_y, cfg)

        val_x = [tr[i] for i in val_idx]
        val_y = y[val_idx]
        val_g = DataLoader(False, val_x, val_y,cfg)

        model = cfg['model'](embed_matrix)
        print(f'load fold{n_fold} weight')
        model.load_weights(f"../model/vqa{n_fold}.h5")
        print('loss                   acc')
        print(model.evaluate_generator(val_g.get_data(),
                                               steps=(val_g.num_samples + bs - 1) // bs,workers=32))

        K.clear_session()
        tf.reset_default_graph()


if __name__ == '__main__':
    #import argparse
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--fold", type=int, required=True)

    #args = parser.parse_args()
    #cfg['fold'] = [args.fold]
    cfg['atten_k'] = 0
    cfg['bs'] = 8
    cfg['seed'] = 47
    cfg['num_frame'] = 16
    cfg['patience'] = 5
    cfg['dim_q'] = 256
    cfg['dim_v'] = 384
    cfg['dim_a'] = 256
    cfg['dim_attr'] = 300
    cfg['attr_num'] = 96
    cfg['use_rc'] = True
    cfg['rc_max_iter'] = 12
    cfg['alpha'] = 0.3

    cfg['attention_avepool'] = True
    cfg['attention'] = 'coa'
    cfg['lr'] = 0.0002
    cfg['loss'] = focal_loss_fixed
    cfg['model'] = attention_model

    main(cfg)




