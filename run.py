import os
import argparse
import random
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from utils.preprocess import Json2Pickle
from utils.preprocess import Pickle2Tsv

from utils.feature import EntityLinkingDataLoader, EntityTypingDataLoader, feature_to_loader
from utils.train import trainer
from model import EntityLinking, EntityTyping



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def main(args):
    set_seed(12345)
    # 数据预处理
    # Json2Pickle().run(args)
    # Pickle2Tsv().run(args)

    if args.model_type == 'entity_linking':
        if args.save_dataloader is False:
            EntityLinkingDataLoader(args).generate_feature()

        train_loader = feature_to_loader(pd.read_pickle(args.feature_path + 'link_loader_train.pkl'), shuffle=True)
        dev_loader = feature_to_loader(pd.read_pickle(args.feature_path + 'link_loader_dev.pkl'), shuffle=False)

        # training
        model = EntityLinking(args)
        trainer(args, model, train_loader, dev_loader)

    else:
        if args.save_dataloader is False:
            EntityTypingDataLoader(args).generate_feature()
        train_loader = feature_to_loader(pd.read_pickle(args.feature_path + 'type_loader_train.pkl'), shuffle=True)
        dev_loader = feature_to_loader(pd.read_pickle(args.feature_path + 'type_loader_dev.pkl'), shuffle=False)

        model = EntityTyping(args)
        trainer(args, model, train_loader, dev_loader)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='entity_linking', help='entity_linking or entity_typing')
    parser.add_argument('--gpu', default=True, help='use_gpu')
    parser.add_argument('--pretrain_path', default='./pretrain_model', help='roberta pretrain')

    # data
    parser.add_argument('--data_path', default='./data/', help='all data')
    parser.add_argument('--raw_path', default='./data/ccks2020_el_data_v1/', help='all data')
    parser.add_argument('--pickle_path', default='./data/pickle/', help='raw to pickle')
    parser.add_argument('--tsv_path', default='./data/tsv/', help='raw to tsv')
    parser.add_argument('--feature_path', default='./data/feature/', help='feature data')
    parser.add_argument('--result_path', default='./data/result/', help='result data')

    parser.add_argument('--pickle_data', default=None, help='pickle data')
    parser.add_argument('--save_dataloader', default=False, help='save_dataloader')

    # save model
    parser.add_argument('--save_el_path', default='./checkpoints/el_models/', help='save entity linking models')
    parser.add_argument('--save_et_path', default='./checkpoints/et_models/', help='save entity typing models')

    # training
    parser.add_argument('--batch_size', default=32, help='batch_size')
    parser.add_argument('--el_max_len', default=384, help='el_max_len')
    parser.add_argument('--et_max_len', default=64, help='et_max_len')
    args = parser.parse_args()


    if not os.path.exists(args.pickle_path):
        os.mkdir(args.pickle_path)
    if not os.path.exists(args.tsv_path):
        os.mkdir(args.tsv_path)
    if not os.path.exists(args.feature_path):
        os.mkdir(args.feature_path)
    if not os.path.exists(args.result_path):
        os.mkdir(args.result_path)

    args.pickle_data = {
        'entity_to_kbids': None,
        'kbid_to_entities': None,
        'kbid_to_text': None,
        'kbid_to_predicates': None,
        'kbid_to_types': None,  # 一个实体可能对应'|'分割的多个类型）
        'idx_to_type': None,
        'type_to_idx': None,
    }

    for k in args.pickle_data:
        pickle_file = args.pickle_path + k + '.pkl'
        if os.path.exists(pickle_file):
            args.pickle_data[k] = pd.read_pickle(pickle_file)
        else:
            print(f'File {pickle_file} not Exist!')

    main(args)


