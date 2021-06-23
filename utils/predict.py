import torch
import torch.nn as nn
import tqdm
import json
import pandas as pd
import numpy as np
from feature import feature_to_loader
from ..model import EntityLinking, EntityTyping


def get_pickle_data():
    pickle_data = {
        'entity_to_kbids': None,
        'kbid_to_entities': None,
        'kbid_to_text': None,
        'kbid_to_predicates': None,
        'kbid_to_types': None,  # 一个实体可能对应'|'分割的多个类型）
        'idx_to_type': None,
        'type_to_idx': None,
    }
    pickle_path = 'data/pickle'
    for k in pickle_data:
        pickle_file = pickle_path + k + '.pkl'
        pickle_data[k] = pd.read_pickle(pickle_file)
    return pickle_data



def predict(model_type, model, test_loader, test_tsv_path, result_tsv_path):
    model.cuda()
    # model = nn.DataParallel(model)
    model.eval()

    if model_type == 'entity_linking':
        result_list, logit_list = [], []
        for batch in tqdm(test_loader):
            for i in range(len(batch)):
                batch[i] = batch[i].cuda()

            input_ids, attention_mask, token_type_ids, labels = batch
            logits = model(input_ids, attention_mask, token_type_ids)
            preds = (logits > 0).int()

            result_list.extend(preds.tolist())
            logit_list.extend(logits.tolist())

        tsv_data = pd.read_csv(test_tsv_path, sep='\t')
        tsv_data['logits'] = logit_list
        tsv_data['result'] = result_list
        tsv_data.to_csv(result_tsv_path, index=False, sep='\t')

    else:
        result_list = []
        for batch in tqdm(test_loader):
            for i in range(len(batch)):
                batch[i] = batch[i].cuda()

            input_ids, attention_mask, token_type_ids, labels = batch
            outputs = model(input_ids, attention_mask, token_type_ids)
            _, preds = torch.max(outputs, dim=1)
            result_list.extend(preds.tolist())
        pickle_data = get_pickle_data()
        idx_to_type = pickle_data['idx_to_type']
        result_list = [idx_to_type[x] for x in result_list]
        tsv_data = pd.read_csv(test_tsv_path, sep='\t')
        tsv_data['result'] = result_list
        tsv_data.to_csv(result_tsv_path, index=False, sep='\t')



def make_predication_result(input_path, output_path, el_res_path, et_res_path):
    pickle_data = get_pickle_data()
    entity_to_kbids = pickle_data['entity_to_kbids']

    el_ret = pd.read_csv(el_res_path, sep='\t',
                         dtype={'text_id': np.str_, 'offset': np.str_, 'kb_id': np.str_})
    et_ret = pd.read_csv(et_res_path, sep='\t',
                         dtype={'text_id': np.str_, 'offset': np.str_})

    result = []
    with open(input_path, 'r') as f:
        for line in tqdm(f):
            line = json.loads(line)
            for data in line['mention_data']:
                text_id = line['text_id']
                offset = data['offset']

                candidate_data = el_ret[(el_ret['text_id'] == text_id) & (el_ret['offset'] == offset)]
                # Entity Linking
                if len(candidate_data) > 0 and candidate_data['logits'].max() > 0:
                    max_idx = candidate_data['logits'].idxmax()
                    data['kb_id'] = candidate_data.loc[max_idx]['kb_id']
                # Entity Typing
                else:
                    type_data = et_ret[(et_ret['text_id'] == text_id) & (et_ret['offset'] == offset)]
                    data['kb_id'] = 'NIL_' + type_data.iloc[0]['result']
            result.append(line)

    with open(output_path, 'w') as f:
        for r in result:
            json.dump(r, f, ensure_ascii=False)
            f.write('\n')


if __name__ == '__main__':
    model_type = 'entity_linking'
    model_path = 'checkpoints/el_models/'
    pickle_data = get_pickle_data()

    if model_type == 'entity_linking':
        test_tsv_path = 'data/tsv/link_test.tsv'
        result_tsv_path = 'data/result/link_test_result.tsv'
        test_loader = feature_to_loader(pd.read_pickle('data/feature/link_loader_test.pkl'), shuffle=False)
        model = EntityLinking.load_from_checkpoint(model_path)
        predict(model_type, model, test_loader, test_tsv_path, result_tsv_path)
    else:
        test_tsv_path = 'data/tsv/type_test.tsv'
        result_tsv_path = 'data/result/type_test_result.tsv'
        test_loader = feature_to_loader(pd.read_pickle('data/feature/type_loader_test.pkl'), shuffle=False)
        model = EntityTyping.load_from_checkpoint(model_path)
        predict(model_type, model, test_loader, test_tsv_path, result_tsv_path)

    # 制作result的json
    input_path = 'data/ccks2020_el_data_v1/test.json'
    output_path = 'data/result/test_result.json'   # 最后的结果文件
    el_res_path = 'data/result/link_test_result.tsv'
    et_res_path = 'data/result/type_test_result.tsv'
    make_predication_result(input_path, output_path, el_res_path, et_res_path)

