from pomegranate import BayesianNetwork
import numpy as np
import pandas as pd
from typing import List
import random
import os
import json
# model_list: List[str] = ['bart-base', 'led-base-16384', 't5-base']
model_list: List[str] = os.listdir('result_multi/bertscore')

def get_data(metrics: str) -> List[int]:
    data = list()
    for i, model in enumerate(model_list):
        file = './result_multi/{}/{}/cnndm/data/article_generation.npy'.format(
            metrics, model)
        model_data = np.load(file, allow_pickle=True)
        for j, length_similar_data in enumerate(model_data):
            for item in length_similar_data:
                new_item = [model] + list(item)
                new_item[1] = j
                new_item[2] = int(item[2]/5)
                new_item[3] = int(item[3]/5)
                new_item[4] = int(new_item[4])
                data.append(new_item)
    return data


def main():
    removed_list = ['compression','chrf','density']
    model_dir = './ba_model'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    random.seed(42)
    metrics_dir = './result_multi'
    metrics_list = os.listdir(metrics_dir)
    for metrics in metrics_list:
        if metrics in removed_list:
            continue
        print(metrics)
        data = get_data(metrics)
        key_list = ['model', 'article length',
                    'groundtruth length', 'generation length', 'score']
        data =  pd.DataFrame(data,columns=key_list)
        edge_list = [(0,4)]
        exclude_edges = [(4,0),(4,1),(4,2),(4,3)]
        ba_model = BayesianNetwork.from_samples(
            data.to_numpy(),state_names = key_list,include_edges=edge_list, exclude_edges=exclude_edges)
        saved_model = ba_model.to_json()
        plt_fir = os.path.join(model_dir,'{}.pdf'.format(metrics))
        model_fir = os.path.join(model_dir,'{}.json'.format(metrics))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        ba_model.plot(plt_fir)
        saved_model = json.loads(saved_model)
        with open(model_fir,mode='w') as fp:
            json.dump(saved_model,fp)
if __name__ == '__main__':
    main()
