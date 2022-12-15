from pomegranate import BayesianNetwork
import numpy as np
import pandas as pd
from typing import List, Tuple
import random
import os
import json
import tqdm
from multiprocessing import Pool
from collections import defaultdict
import sys
import matplotlib.pyplot as plt

model_list: List[str] = ['bart-base', 'led-base-16384', 't5-base']
dataset_list = ['cnn_dailymail','wikihowAll','corpus-webis-tldr-17']
score_type_list = ['percent','value','original-score', 'predict-score']
metrics_list=['rouge_1_f_score', 'rouge_2_f_score', 'rouge_3_f_score','rouge_4_f_score', 'rouge_l_f_score', 'rouge_su*_f_score', 'rouge_w_1.2_f_score',  'rouge_we_3_f', 'bert_score_f1', 'mover_score', 'blanc', 'supert', 'bleu', 'chrf', 'meteor']
names_list=['ROUGE-1', 'ROUGE-2', 'ROUGE-3', 'ROUGE-4', 'ROUGE-L','ROUGE-su*', 'ROUGE-w', 'ROUGE-we-3','BertScore-f', 'MoverScore', 'BLANC', 'SuPERT', 'BLEU', 'CHRF', 'METEOR',]
color_dict = {'ROUGE-1':'red','ROUGE-2':'darkred','ROUGE-3':'firebrick','ROUGE-4':'brown','ROUGE-L':'indianred','ROUGE-su*':'lightcoral','ROUGE-w':'tomato','ROUGE-we-3':'yellow','BertScore-f':'cyan','MoverScore':'aquamarine','BLANC':'lime','SuPERT':'green','BLEU':'b','CHRF':'royalblue','METEOR':'violet'}

# model_list: List[str] = ['bart-base', 'led-base-16384', 't5-base']
# dataset_list = ['cnn_dailymail','wikihowAll']
# score_type_list = ['percent']
# metrics_list=['bert_score_f1']
# names_list=['BertScore-f']
# color_dict = {'BertScore-f':'cyan'}


def get_data_size(metrics: str, dataset:str, model:str, random_mark:bool) -> Tuple[List[int],List[int]]:
    split_number = 10
    different_number = 10

    
    data = list()
    file = './result/{}/{}/{}/data/article_generation.npy'.format(
        metrics,model,dataset)
    if random_mark:
        file = file.replace('article_generation.npy','article_generation_random.npy')
    model_data = np.load(file, allow_pickle=True)
    full_data = list()
    for item in model_data:
        full_data.extend(item)
    for i, item in enumerate(full_data):
        full_data[i] = list(item)
    random.shuffle(full_data)

    sorted_full_data = sorted(full_data, key=lambda x: x[0])
    index_count = int(len(sorted_full_data) / split_number)
    for index, item in enumerate(sorted_full_data):
        item[0] = min(int(index / index_count),split_number-1)
    random.shuffle(full_data)

    sorted_full_data = sorted(full_data, key=lambda x: x[1])
    index_count = int(len(sorted_full_data) / split_number)
    for index, item in enumerate(sorted_full_data):
        item[1] = min(int(index / index_count),split_number-1)
    random.shuffle(full_data)

    splited_data = [[[] for _ in range(split_number)] for _ in range(split_number)]
    for item in full_data:
        splited_data[item[0]][item[1]].append(item)
    for article_index in range(split_number):
        for reference_index in range(split_number):
            splited_item = splited_data[article_index][reference_index]
            if len(splited_item)<different_number:
                continue
            random.shuffle(splited_item)
            sorted_splited_item = sorted(splited_item, key=lambda x: x[2])
            index_count = int(len(sorted_splited_item) / different_number)
            for index, item in enumerate(sorted_splited_item):
                item[2] = min(int(index / index_count),different_number-1)

    generation_length_list = set()            
    for article_index in range(split_number):
        for reference_index in range(split_number):
            for item in splited_data[article_index][reference_index]:
                generation_length_list.add(item[2])
                if item[-1] > 100:
                    item[-1] /= 100
                if metrics in ['bert_score_f1', 'blanc']:
                    item[-1] = item[-1] + 100
                item[-1] = int(10*item[-1])
                data.append(item)

    return data, list(generation_length_list)


def get_data_width(metrics: str, dataset:str, model:str, random_mark:bool) -> Tuple[List[int],List[int]]:
    artilce_split_length = 10
    reference_split_length = 10
    summary_split_length = 10

    data = list()

    file = './result/{}/{}/{}/data/article_generation.npy'.format(
        metrics,model,dataset)
    if random_mark:
        file = file.replace('article_generation.npy','article_generation_random.npy')
    model_data = np.load(file, allow_pickle=True)
    for j, length_similar_data in enumerate(model_data):
        temp_item_list = list()
        for item in length_similar_data:
            temp_item_list.append(list(item))
        random.shuffle(temp_item_list)
        sorted_item_list = temp_item_list

        # sorted_item_list = sorted(temp_item_list, key=lambda x: x[2])
        # index_count = int (len(sorted_item_list) / 10)
        # for index, item in enumerate(sorted_item_list):
        #     item[2] = min(int(index / index_count),9)

        # for index, item in enumerate(sorted_item_list):
        #     item[2] = int(item[2] / 5)       
        # random.shuffle(temp_item_list)

        # sorted_item_list = sorted(temp_item_list, key=lambda x: x[1])
        # index_count = int (len(sorted_item_list) / 10)
        # for index, item in enumerate(sorted_item_list):
        #     item[1] = min(int(index / index_count),9)

        for item in sorted_item_list:
            item[0] = int(item[0]/artilce_split_length)
            item[1] = int(item[1]/reference_split_length)
            item[2] = int(item[2]/summary_split_length)
            if item[3] > 100:
                item[3] /= 100
            if metrics in ['bert_score_f1', 'blanc']:
                item[-1] = item[-1] + 100
            item[3] = int(item[3]*10)
            # item[3] = int(item[3])

        data.extend(sorted_item_list)
        generation_length_list = set()
        for item in data:
            generation_length_list.add(item[2])
    return data, list(generation_length_list)


def get_data(divided_type:str, metrics: str, dataset:str, model:str, random_mark:bool) -> Tuple[List[int],List[int]]:
    if divided_type == 'width':
        return get_data_width(metrics, dataset, model, random_mark)
    elif divided_type == 'size':
        return get_data_size(metrics, dataset, model, random_mark)

def calculate_score(score_type_list,divided_type, model_type, metrics, random_mark, dataset, model):
    score_list = dict()
    for score_type in score_type_list:
        score_list[score_type] = list()
    # predict_score_list = list()
    try:
        data, generation_length_list = get_data(divided_type, metrics, dataset, model, random_mark)
    except:
        file = './result/{}/{}/{}/data/article_generation.npy'.format(
            metrics,model,dataset)
        if random_mark:
            file = file.replace('article_generation.npy','article_generation_random.npy')
        print(file)
    key_list = ['article length',
                    'groundtruth length', 'generation length', 'score']
    data =  pd.DataFrame(data,columns=key_list)
    edge_list = [(0,3),(2,3)]
    # edge_list = [(2,3)]
    exclude_edges = [(3,0),(3,1),(3,2),(2,0),(2,1),(1,0)]
    ba_model = BayesianNetwork.from_samples(
            data.to_numpy(),state_names = key_list,include_edges=edge_list, exclude_edges=exclude_edges)

    if metrics == 'rouge_w_1.2_f_score':
        metric = 'rouge_w_12_f_score'
    else:
        metric = metrics    

    calculate_generation_length = list()
    for i, generation_length in enumerate(generation_length_list[:-1]):
    # for i, generation_length in enumerate(generation_length_list[:5])
        splited_data = data.loc[data['generation length']==generation_length].to_numpy()
        if len(splited_data) <= 5:
                    continue
        if dataset == 'corpus-webis-tldr-17':
            sample_list = [i for i in range(len(splited_data))]
            sample_list = random.sample(sample_list, max(int(len(sample_list)/10),min(50,len(sample_list))))
            splited_data = splited_data[sample_list,:]
        calculate_generation_length.append(generation_length*10)
        original_score = np.mean(splited_data,axis=0)[-1]
        splited_data = splited_data.tolist()
        for item in splited_data:
            item[-1] = None
            item[2] = generation_length_list[i+1]
        predicted_prob = ba_model.predict_proba(splited_data)
        predictd_result = list()
        for item in predicted_prob:
            score = 0
            for k,v in item[-1].parameters[0].items():
                if v !=0:
                    score += k*v
            predictd_result.append(score)
        predict_score = np.mean(predictd_result,axis=0)
        difference_value = predict_score - original_score
        difference_percent = (difference_value / max(original_score,1))*100
        if metrics in ['bert_score_f1', 'blanc']:
            original_score -= 1000
            predict_score -= 1000
        if 'value' in score_type_list:
            score_list['value'].append(difference_value/10)
        if 'percent' in score_type_list:
            score_list['percent'].append(difference_percent)
        if 'original-score' in score_type_list:
            score_list['original-score'].append(original_score/10)
        if 'predict-score' in score_type_list:
            score_list['predict-score'].append(predict_score/10)

    score = dict()
    for score_type, s_list in score_list.items():
        score[score_type] = np.mean(s_list)
    
    return (metrics, random_mark, dataset, model, score, score_list, calculate_generation_length)


def draw_picture(score_type_list, divided_type, mode_type, model, random_info, picture_into):
    for score_type in score_type_list:
        fig_dir = './ba_model/picture/{}/{}/{}/'.format(score_type,divided_type,mode_type)
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        picture_file = os.path.join(fig_dir, '{}_{}.pdf'.format(model, random_info))
        fig = plt.figure(figsize=(22, 6))
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.15, hspace=0)
        subplot_index = 1
        for dataset, dataset_data in picture_into.items(): 
            ax = fig.add_subplot(13*10+subplot_index)
            for item_info, item in dataset_data.items():
                # plt.yscale('log')
                # plt.yscale('symlog', linthresh=1.0, linscale=10)
                ax.plot(item['generation_length'],
                        item['score'][score_type], label=item_info, linewidth=3, color=color_dict[item_info])
            ax.set_title(dataset, fontsize=20, weight='bold')
            for label in ax.get_xticklabels():
                label.set_fontproperties({'weight': 'bold', 'size': 20})
            for label in ax.get_yticklabels():
                label.set_fontproperties({'weight': 'bold', 'size': 20})
            subplot_index += 1
        fig.suptitle("The performance change with differnet generated summary length", fontsize=20, weight='bold')

        fig.legend(list(color_dict.keys()), loc='upper center',
                bbox_to_anchor=(0.5, 0),  ncol=5, prop={'weight': 'bold', 'size': 20} )
        fig.supxlabel("the length of generated summary",
                    fontsize=20, weight='bold')
        fig.savefig(picture_file, bbox_inches='tight', pad_inches=0)
        plt.close()

def main():

    divided_type = sys.argv[1]
    model_type = sys.argv[2]
    # divided_type = 'width'

    pbar = tqdm.tqdm(total=len(metrics_list) * len(model_list)*len(dataset_list)*2, ncols=100)
    update = lambda *args: pbar.update()

    random.seed(42)
    res_list = list()

    conclusion = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(float))))
    pict_info = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(list())))))

    with Pool(processes=18) as pool:
        for metrics in metrics_list:
            for random_mark in [False, True]:
                for dataset in dataset_list:
                    for model in model_list:
                        res_list.append(pool.apply_async(calculate_score,(score_type_list,divided_type, model_type, metrics,random_mark,dataset,model), callback=update))
        pool.close()
        pool.join()
    for res in res_list:
        metrics,random_mark,dataset,model,score, score_list, generation_length_list = res.get()
        metrics = names_list[metrics_list.index(metrics)]
        conclusion[metrics][random_mark][dataset][model] = score
        pict_info[model][random_mark][dataset][metrics] = {'score':score_list, 'generation_length':generation_length_list}



    # for metrics in metrics_list:
    #     for random_mark in [False, True]:
    #         for dataset in dataset_list:
    #             for model in model_list:
    #                 res_list.append(calculate_score(score_type_list,divided_type, metrics,random_mark,dataset,model))
    #                 pbar.update()
    # for res in res_list:
    #     metrics,random_mark,dataset,model,score, score_list, generation_length_list = res
    #     metrics = names_list[metrics_list.index(metrics)]
    #     conclusion[metrics][random_mark][dataset][model] = score
    #     pict_info[model][random_mark][dataset][metrics] = {'score':score_list, 'generation_length':generation_length_list}

    for model, model_data in pict_info.items():
        for random_info, random_data in model_data.items():
            draw_picture(score_type_list, divided_type, model_type, model, random_info, random_data)

    pbar.close()
    for score_type in score_type_list:
        if not os.path.exists('./paper/bayesian/{}/{}'.format(score_type, model_type)):
            os.makedirs('./paper/bayesian/{}/{}'.format(score_type, model_type))
        with open('./paper/bayesian/{}/{}/{}_test.csv'.format(score_type, model_type,divided_type),mode='w',encoding='utf8') as fp:
            title = ','.join(model_list)
            title = title+','+'mean'
            fp.write('Metrics, {},{},{},{},{},{}'.format(title,title,title,title,title,title)+'\n')
            for metric in names_list:
                item = metric
                for random_mark in [False, True]:
                    for dataset in dataset_list:
                        mean_list = list()
                        for model in model_list:
                            item = item +',{:.2f}'.format(conclusion[metric][random_mark][dataset][model][score_type])
                            mean_list.append(conclusion[metric][random_mark][dataset][model][score_type])
                        mean_score = np.mean(mean_list)
                        consistent_mark = True
                        for score in mean_list:
                            if mean_score * score < 0:
                                consistent_mark = False
                                break
                        if not consistent_mark:
                            mean_score = "{:.2f}*".format(mean_score)
                        else:
                            mean_score = "{:.2f}".format(mean_score)
                        item = item + ','+ mean_score
                fp.write(item+'\n')


if __name__ == '__main__':
    main()
