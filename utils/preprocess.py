from collections import defaultdict
import pandas as pd
import random
import tqdm
import json
import torch



class Json2Pickle:

    def __init__(self):
        self.entity_to_kbids = defaultdict(set)  # 每个实体对应的kbid, 1对多
        self.kbid_to_entities = dict()           # kbid对应实体列表， 1对多
        self.kbid_to_text = dict()               # kbid对应的文本
        self.kbid_to_predicates = dict()         # kbid对应的属性
        self.kbid_to_types = dict()              # kbid对应的类型列表
        self.idx_to_type = list()                # id对应类型
        self.type_to_idx = dict()                # 类型对应id


    def run(self, args, shuffle_text=True):
        with open(args.raw_path + 'kb.json', 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                line = json.loads(line)

                kbid = line['subject_id']  # '10001'
                # 将实体名与别名合并
                entities = set(line['alias'])
                entities.add(line['subject'])
                for entity in entities:
                    self.entity_to_kbids[entity].add(kbid)
                self.kbid_to_entities[kbid] = entities
                text_list, predicate_list = [], []
                for x in line['data']:
                    # 简单拼接predicate与object，这部分可以考虑别的方法尝试
                    text_list.append(':'.join([x['predicate'].strip(), x['object'].strip()]))
                    predicate_list.append(x['predicate'].strip())
                if shuffle_text:  # 对属性文本随机打乱顺序
                    random.shuffle(text_list)
                self.kbid_to_predicates[kbid] = predicate_list
                self.kbid_to_text[kbid] = ' '.join(text_list)
                # 删除文本中的特殊字符
                for c in ['\r', '\t', '\n']:
                    self.kbid_to_text[kbid] = self.kbid_to_text[kbid].replace(c, '')

                type_list = line['type'].split('|')
                self.kbid_to_types[kbid] = type_list
                for t in type_list:
                    if t not in self.type_to_idx:
                        self.type_to_idx[t] = len(self.idx_to_type)
                        self.idx_to_type.append(t)

        pd.to_pickle(self.entity_to_kbids, args.pickle_path + 'entity_to_kbids.pkl')
        pd.to_pickle(self.kbid_to_entities, args.pickle_path + 'kbid_to_entities.pkl')
        pd.to_pickle(self.kbid_to_text, args.pickle_path + 'kbid_to_text.pkl')
        pd.to_pickle(self.kbid_to_predicates, args.pickle_path + 'kbid_to_predicates.pkl')
        pd.to_pickle(self.kbid_to_types, args.pickle_path + 'kbid_to_types.pkl')
        pd.to_pickle(self.idx_to_type, args.pickle_path + 'idx_to_type.pkl')
        pd.to_pickle(self.type_to_idx, args.pickle_path + 'type_to_idx.pkl')
        print('Process Pickle File Finish.')


class Pickle2Tsv:
    """生成模型训练、验证、推断所需的tsv文件"""
    def __init__(self):
        pass

    def process_link_data(self, args, input_path, output_path, max_negs=-1):
        entity_to_kbids = args.pickle_data['entity_to_kbids']
        kbid_to_text = args.pickle_data['kbid_to_text']
        kbid_to_predicates = args.pickle_data['kbid_to_predicates']
        link_dict = defaultdict(list)

        with open(input_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                line = json.loads(line)

                for data in line['mention_data']:
                    # 对测试集特殊处理
                    if 'kb_id' not in data:
                        data['kb_id'] = '0'

                    # KB中不存在的实体不进行链接
                    if not data['kb_id'].isdigit():
                        continue

                    entity = data['mention']
                    kbids = list(entity_to_kbids[entity])
                    random.shuffle(kbids)

                    num_negs = 0
                    for kbid in kbids:
                        if num_negs >= max_negs > 0 and kbid != data['kb_id']:
                            continue

                        link_dict['text_id'].append(line['text_id'])
                        link_dict['entity'].append(entity)
                        link_dict['offset'].append(data['offset'])
                        link_dict['short_text'].append(line['text'])
                        link_dict['kb_id'].append(kbid)
                        link_dict['kb_text'].append(kbid_to_text[kbid])
                        link_dict['kb_predicate_num'].append(len(kbid_to_predicates[kbid]))
                        if kbid != data['kb_id']:
                            link_dict['predict'].append(0)
                            num_negs += 1
                        else:
                            link_dict['predict'].append(1)
        link_data = pd.DataFrame(link_dict)
        link_data.to_csv(output_path, index=False, sep='\t')


    def process_type_data(self, args, input_path, output_path):
        kbid_to_types = args.pickle_data['kbid_to_types']
        type_dict = defaultdict(list)

        with open(input_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                line = json.loads(line)

                for data in line['mention_data']:
                    entity = data['mention']

                    # 测试集特殊处理
                    if 'kb_id' not in data:
                        entity_type = ['Other']
                    elif data['kb_id'].isdigit():
                        entity_type = kbid_to_types[data['kb_id']]
                    else:
                        entity_type = data['kb_id'].split('|')
                        for x in range(len(entity_type)):
                            entity_type[x] = entity_type[x][4:]
                    for e in entity_type:
                        type_dict['text_id'].append(line['text_id'])
                        type_dict['entity'].append(entity)
                        type_dict['offset'].append(data['offset'])
                        type_dict['short_text'].append(line['text'])
                        type_dict['type'].append(e)

        type_data = pd.DataFrame(type_dict)
        type_data.to_csv(output_path, index=False, sep='\t')


    def run(self, args):
        self.process_link_data(args, input_path=args.raw_path + 'train.json',
                               output_path=args.tsv_path + 'link_train.tsv', max_negs=2)
        self.process_link_data(args, input_path=args.raw_path + 'dev.json',
                               output_path=args.tsv_path + 'link_dev.tsv', max_negs=-1)
        self.process_link_data(args, input_path=args.raw_path + 'test.json',
                               output_path=args.tsv_path + 'link_test.tsv', max_negs=-1)

        self.process_type_data(args, input_path=args.raw_path + 'train.json',
                               output_path=args.tsv_path + 'type_train.tsv')
        self.process_type_data(args, input_path=args.raw_path + 'dev.json',
                               output_path=args.tsv_path + 'type_dev.tsv')
        self.process_type_data(args, input_path=args.raw_path + 'test.json',
                               output_path=args.tsv_path + 'type_test.tsv')

