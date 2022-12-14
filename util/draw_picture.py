import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import DefaultDict, List
import random


def draw_picture(data: DefaultDict[str, DefaultDict[str, DefaultDict[str,List[float]]]],control_factor:str, x_factor:str, based_type:str, based_info:str, random_info) -> None:
    fig_dir = './picture/{}/{}/{}/{}/{}'.format(based_type, based_info, random_info,control_factor,x_factor)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    for base, base_data in data.items():
        if base == 'rouge_w_1.2_f_score':
            base = 'rouge_w_12_f_score'
        picture_file = os.path.join(fig_dir,'{}'.format(base))
        for item_info, item in base_data.items():
            plt.plot([len(item['mean'])*j for j in range(len(item['mean']))], item['mean'], label=item_info)
        plt.title("The performance when controlling {} length".format(control_factor))
        plt.legend(bbox_to_anchor=(1.05, 1))
        plt.xlabel("percent of length of {}".format(x_factor))
        plt.savefig(picture_file,bbox_inches='tight')
        plt.close()


def get_result(bucket_data, split_base):
    split_number = 10
    split_index = 0
    if split_base == 'article':
        split_index = 0
    elif split_base == 'groundtruth':
        split_index = 1
    elif split_base == 'generation':
        split_index = 2

    total_bucket = [[] for _ in range(split_number)]
    for _, single_bucket in enumerate(bucket_data):
        sorted_bucket_data = [list(it) for it in single_bucket]
        random.shuffle(sorted_bucket_data)
        sorted_bucket_data = sorted(sorted_bucket_data , key = lambda x:x[split_index])
        split_count = int (len(sorted_bucket_data) / 10)
        for index, item in enumerate(sorted_bucket_data):
            item[split_index] = min(int(index / split_count),9)

        bucket = [[] for _ in range(split_number)]
        for j in range(split_number):
            bucket[j] = [it[-1]
                         for it in sorted_bucket_data[j*split_count:(j+1)*split_count]]
        if split_count * split_number != len(bucket):
            for it in sorted_bucket_data[split_number*split_count:]:
                bucket[j].append(it[-1])
        mean = [np.mean(d) for d in bucket]
        # mean = [np.mean(d) for d in bucket]
        # diff_mean = mean - np.min(mean)
        for i, score in enumerate(mean):
            total_bucket[i].append(score)


    mean_result = [np.mean(d) for d in total_bucket]
    std_result = [np.std(d) for d in total_bucket]
    result = {'mean':mean_result,'std':std_result}
    return result


def draw_result_picture(metrics_list:List[str], model_list:List[str], dataset_list:List[str], control_factor:str, x_factor:str) -> None:
    for dataset in dataset_list:
        result:  DefaultDict[str, DefaultDict[str, DefaultDict[str,List[float]]]] = defaultdict(lambda: defaultdict(lambda: dict))
        for metrics in metrics_list:
            for model in model_list:
                file = './result/{}/{}/{}/data/{}_{}.npy'.format(
                    metrics, model,dataset,control_factor, x_factor)
                item = np.load(file, allow_pickle=True)
                item = get_result(item, x_factor)
                result[metrics][model] = item
        draw_picture(result,control_factor,x_factor,'dataset',dataset,'prediction')
    for metrics in metrics_list:
        result:  DefaultDict[str, DefaultDict[str, DefaultDict[str,List[float]]]] = defaultdict(lambda: defaultdict(lambda: dict))
        for model in model_list:
            for dataset in dataset_list:
                file = './result/{}/{}/{}/data/{}_{}.npy'.format(
                    metrics, model,dataset,control_factor, x_factor)
                item = np.load(file, allow_pickle=True)
                item = get_result(item, x_factor)
                result[model][dataset] = item
        draw_picture(result,control_factor,x_factor,'metrics',metrics,'prediction')



def draw_random_picture(metrics_list:List[str], model_list:List[str], dataset_list:List[str], control_factor:str, x_factor:str) -> None:
    for dataset in dataset_list:
        result:  DefaultDict[str, DefaultDict[str, DefaultDict[str,List[float]]]] = defaultdict(lambda: defaultdict(lambda: dict))
        for metrics in metrics_list:
            for model in model_list:
                file = './result/{}/{}/{}/data/{}_{}_random.npy'.format(
                    metrics, model,dataset,control_factor, x_factor)
                item = np.load(file, allow_pickle=True)
                item = get_result(item, x_factor)
                result[metrics][model] = item
        draw_picture(result,control_factor,x_factor,'dataset',dataset,'random')
    for metrics in metrics_list:
        result:  DefaultDict[str, DefaultDict[str, DefaultDict[str,List[float]]]] = defaultdict(lambda: defaultdict(lambda: dict))
        for model in model_list:
            for dataset in dataset_list:
                file = './result/{}/{}/{}/data/{}_{}_random.npy'.format(
                    metrics, model,dataset,control_factor, x_factor)
                item = np.load(file, allow_pickle=True)
                item = get_result(item, x_factor)
                result[model][dataset] = item
        draw_picture(result,control_factor,x_factor,'metrics',metrics,'random')

def draw_groundtruth_picture(metrics_list:List[str], model_list:List[str], dataset_list:List[str], control_factor:str, x_factor:str) -> None:
    for dataset in dataset_list:
        result:  DefaultDict[str, DefaultDict[str, DefaultDict[str,List[float]]]] = defaultdict(lambda: defaultdict(lambda: dict))
        for metrics in metrics_list:
            for model in model_list:
                file = './result/{}/{}/{}/data/{}_{}_random.npy'.format(
                    metrics, model,dataset,control_factor, x_factor)
                item = np.load(file, allow_pickle=True)
                item = get_result(item, x_factor)
                result[metrics][model] = item
        draw_picture(result,control_factor,x_factor,'dataset',dataset,'groundtruth')
    for metrics in metrics_list:
        result:  DefaultDict[str, DefaultDict[str, DefaultDict[str,List[float]]]] = defaultdict(lambda: defaultdict(lambda: dict))
        for model in model_list:
            for dataset in dataset_list:
                file = './result/{}/{}/{}/data/{}_{}_random.npy'.format(
                    metrics, model,dataset,control_factor, x_factor)
                item = np.load(file, allow_pickle=True)
                item = get_result(item, x_factor)
                result[model][dataset] = item
        draw_picture(result,control_factor,x_factor,'metrics',metrics,'random')

def main():
    random.seed(42)
    metrics_list = ['rouge_1_f_score', 'rouge_2_f_score', 'rouge_3_f_score','rouge_4_f_score', 'rouge_l_f_score', 'rouge_su*_f_score', "rouge_w_1.2_f_score",  'rouge_we_3_f', 'bert_score_f1', 'mover_score', 'blanc', 'supert', 'bleu', 'chrf', 'meteor', 'summary_length', 'percentage_novel_1-gram','percentage_novel_2-gram', 'percentage_novel_3-gram','percentage_repeated_1-gram_in_summ', 'percentage_repeated_2-gram_in_summ','percentage_repeated_3-gram_in_summ', 'coverage', 'compression', 'density']
    # metrics_list = ['bert_score_f1','blanc','mover_score','supert']
    model_list = ['bart-base', 'led-base-16384', 't5-base']
    dataset_list = ['cnn_dailymail', 'wikihowAll']
    factor_list = ['article','generation','groundtruth']
    draw_groundtruth_picture(metrics_list,['groundtruth'],['cnn_dailymail'], 'article', 'generation')
    for control_factor in factor_list:
        for x_factor in factor_list:
            if control_factor != x_factor:
                draw_result_picture(metrics_list, model_list,
                                    dataset_list, control_factor, x_factor)
                draw_random_picture(metrics_list, model_list,
                                    dataset_list, control_factor, x_factor)


if __name__ == '__main__':
    main()
