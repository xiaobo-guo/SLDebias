import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import DefaultDict, List
import random
import math

# color_dict = {'rouge_1_f_score':'red','rouge_2_f_score':'darkred','rouge_3_f_score':'firebrick','rouge_4_f_score':'brown','rouge_l_f_score':'indianred','rouge_su*_f_score':'lightcoral','rouge_w_1.2_f_score':'tomato','rouge_we_3_f':'yellow','bert_score_f1':'cyan','mover_score':'aquamarine','blanc':'lime','supert':'green','bleu':'b','chrf':'royalblue','meteor':'violet'}
metrics_list = ['bert_score_f1', 'blanc', 'bleu', 'chrf', 'meteor', 'mover_score', 'rouge_1_f_score', 'rouge_2_f_score',
                'rouge_3_f_score', 'rouge_4_f_score', 'rouge_l_f_score', 'rouge_su*_f_score', 'rouge_w_1.2_f_score',  'rouge_we_3_f']
names_list = ['BertScore', 'BLANC', 'BLEU', 'chrF', 'METEOR', 'MoverScore', 'ROUGE-1',
              'ROUGE-2', 'ROUGE-3', 'ROUGE-4', 'ROUGE-L', 'ROUGE-su*', 'ROUGE-w', 'ROUGE-we-3']
color_dict = {'ROUGE-1': 'rosybrown', 'ROUGE-2': 'brown', 'ROUGE-3': 'red', 'ROUGE-4': 'mistyrose', 'ROUGE-L': 'sienna', 'ROUGE-su*': 'lightcoral', 'ROUGE-w': 'tomato',
              'ROUGE-we-3': 'darkorange', 'BertScore': 'cyan', 'MoverScore': 'deepskyblue', 'BLANC': 'lime', 'BLEU': 'b', 'chrF': 'deeppink', 'METEOR': 'violet'}
factor_to_file_dict = {'article': 'article',
                       'reference': 'groundtruth', 'generated summary': 'generation'}
dataset_name_dict = {'cnn_dailymail': 'CNN/DM',
                     'wikihowAll': 'WikiHow', 'corpus-webis-tldr-17': 'Web-tldr'}


def draw_picture(data: DefaultDict[str, DefaultDict[str, DefaultDict[str, List[float]]]], control_factor_list: List[str], x_factor: str, based_type: str, based_info: str, random_info) -> None:
    fig_dir = './picture/{}/{}/{}/'.format(
        based_type, based_info, random_info)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    picture_file = os.path.join(fig_dir, '{}_{}_{}_{}'.format(
        '-'.join(control_factor_list), x_factor, random_info, based_info))
    fig = plt.figure(figsize=(22, 6))
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.15, hspace=0)
    subplot_index = 1
    for base, base_data in data.items():
        ax = fig.add_subplot(13*10+subplot_index)
        for item_info, item in base_data.items():
            # plt.yscale('log')
            # plt.yscale('symlog', linthresh=1.0, linscale=10)
            # ax.plot(['{}%'.format(int(100 / len(item['mean'])*j)) for j in range(len(item['mean']))],item['mean'], label=names_list[metrics_list.index(item_info)], linewidth=3, color=color_dict[names_list[metrics_list.index(item_info)]],)
            x_labels = ['{}%'.format(int(100 / len(item['mean'])*j)) for j in range(len(item['mean']))],item['mean']
            ax.plot(item['mean'],label=names_list[metrics_list.index(item_info)], linewidth=3, color=color_dict[names_list[metrics_list.index(item_info)]],)
            plt.xticks([0,4,8,12,16,19],['0%','20%','40%','60%','80%','95%'])
        ax.set_title(dataset_name_dict[base], fontsize=20, weight='bold')
        for label in ax.get_xticklabels():
            label.set_fontproperties({'weight': 'bold', 'size': 20})
        for label in ax.get_yticklabels():
            label.set_fontproperties({'weight': 'bold', 'size': 20})
        subplot_index += 1
    fig.suptitle("The change of reported score with various percentile of {} length".format(
        x_factor), fontsize=20, weight='bold')

    fig.legend(names_list, loc='upper center',
               bbox_to_anchor=(0.5, 0),  ncol=7,columnspacing=0.7, handletextpad=0.3, prop={'weight': 'bold', 'size': 20})
    fig.supxlabel("percentile of {} length".format(x_factor),
                  fontsize=20, weight='bold')
    fig.savefig(picture_file+'.pdf', bbox_inches='tight', pad_inches=0)
    plt.close()


def get_result(bucket_data, control_factor_list: List[str], x_factor: str, metric: str):
    factor_to_index_list = ['article', 'reference', 'generated summary']
    control_factor_index_list = list()
    split_number = 10
    picture_number = 20

    full_data = list()
    for item in bucket_data:
        full_data.extend(item)
    for i, item in enumerate(full_data):
        item_list = list(item)
        if item_list[-1] > 100:
            item_list[-1] /= 100
        full_data[i] = item_list
    random.shuffle(full_data)

    sorted_full_data = full_data
    if metric in ['bert_score_f1', 'blanc']:
        for i, item in enumerate(sorted_full_data):
            sorted_full_data[i][-1] += 100
    for control_factor in control_factor_list:
        control_index = factor_to_index_list.index(control_factor)
        control_factor_index_list.append(control_index)
        sorted_full_data = sorted(
            sorted_full_data, key=lambda x: x[control_index])
        index_count = int(len(sorted_full_data) / split_number)
        for index, item in enumerate(sorted_full_data):
            item[control_index] = min(int(index / index_count), split_number-1)
        random.shuffle(sorted_full_data)

    full_data = sorted_full_data
    if len(control_factor_list) == 2:
        total_bucket = [[] for _ in range(picture_number)]
        splited_data = [[[] for _ in range(split_number)]
                        for _ in range(split_number)]
        for item in full_data:
            splited_data[item[control_factor_index_list[0]]
                         ][item[control_factor_index_list[1]]].append(item)

        xfactor_index = factor_to_index_list.index(x_factor)
        for article_index in range(split_number):
            for groundtruth_index in range(split_number):
                splited_item = splited_data[article_index][groundtruth_index]
                if len(splited_item) < picture_number:
                    continue
                random.shuffle(splited_item)
                sorted_splited_item = sorted(
                    splited_item, key=lambda x: x[xfactor_index])
                index_count = int(len(sorted_splited_item) / picture_number)
                for index, item in enumerate(sorted_splited_item):
                    total_bucket[min(int(index / index_count),
                                     picture_number-1)].append(item[-1])
    else:
        total_bucket = [[] for _ in range(picture_number)]
        splited_data = [[] for _ in range(split_number)]
        for item in full_data:
            splited_data[item[control_factor_index_list[0]]].append(item)

        xfactor_index = factor_to_index_list.index(x_factor)
        for control_index in range(split_number):
            splited_item = splited_data[control_index]
            if len(splited_item) < picture_number:
                continue
            random.shuffle(splited_item)
            sorted_splited_item = sorted(
                splited_item, key=lambda x: x[xfactor_index])
            index_count = int(len(sorted_splited_item) / picture_number)
            for index, item in enumerate(sorted_splited_item):
                total_bucket[min(int(index / index_count),
                                 picture_number-1)].append(item[-1])

    mean_result = [np.mean(d) for d in total_bucket]
    mean_result = np.array(mean_result) - np.min(mean_result)
    # mean_result = np.log(100*(np.array(mean_result) -
    #                      np.min(mean_result))/np.min(mean_result))
    for i, item in enumerate(mean_result):
        if item == -np.inf:
            mean_result[i] = 0
    # mean_result = (np.array(mean_result) - np.min(mean_result))
    std_result = [np.std(d) for d in total_bucket]
    result = {'mean': mean_result, 'std': std_result}
    return result


def draw_result_picture(metrics_list: List[str], model_list: List[str], dataset_list: List[str], control_factor_list: str, x_factor: str) -> None:
    # for dataset in dataset_list:
    #     result:  DefaultDict[str, DefaultDict[str, DefaultDict[str,List[float]]]] = defaultdict(lambda: defaultdict(lambda: dict))
    #     for metrics in metrics_list:
    #         for model in model_list:
    #             file = './result/{}/{}/{}/data/{}_{}.npy'.format(
    #                 metrics, model,dataset,control_factor, x_factor)
    #             item = np.load(file, allow_pickle=True)
    #             item = get_result(item, x_factor)
    #             result[metrics][model] = item
    #     draw_picture(result,control_factor,x_factor,'dataset',dataset,'prediction')
    # for metrics in metrics_list:
    #     result:  DefaultDict[str, DefaultDict[str, DefaultDict[str,List[float]]]] = defaultdict(lambda: defaultdict(lambda: dict))
    #     for model in model_list:
    #         for dataset in dataset_list:
    #             file = './result/{}/{}/{}/data/{}_{}.npy'.format(
    #                 metrics, model,dataset,control_factor, x_factor)
    #             item = np.load(file, allow_pickle=True)
    #             item = get_result(item, x_factor)
    #             result[model][dataset] = item
    #     draw_picture(result,control_factor,x_factor,'metrics',metrics,'prediction')

    for model in model_list:
        result:  DefaultDict[str, DefaultDict[str, DefaultDict[str, List[float]]]] = defaultdict(
            lambda: defaultdict(lambda: dict))

        control_factor = control_factor_list[0]
        for dataset in dataset_list:
            for metrics in metrics_list:
                file = './result/{}/{}/{}/data/{}_{}.npy'.format(
                    metrics, model, dataset, factor_to_file_dict[control_factor], factor_to_file_dict[x_factor])
                item = np.load(file, allow_pickle=True)
                item = get_result(item, control_factor_list, x_factor, metrics)
                result[dataset][metrics] = item
        draw_picture(result, control_factor_list, x_factor,
                     'model', model, 'prediction')


def draw_random_picture(metrics_list: List[str], model_list: List[str], dataset_list: List[str], control_factor_list: str, x_factor: str) -> None:
    # for dataset in dataset_list:
    #     result:  DefaultDict[str, DefaultDict[str, DefaultDict[str,List[float]]]] = defaultdict(lambda: defaultdict(lambda: dict))
    #     for metrics in metrics_list:
    #         for model in model_list:
    #             file = './result/{}/{}/{}/data/{}_{}_random.npy'.format(
    #                 metrics, model,dataset,control_factor, x_factor)
    #             item = np.load(file, allow_pickle=True)
    #             item = get_result(item, x_factor)
    #             result[metrics][model] = item
    #     draw_picture(result,control_factor,x_factor,'dataset',dataset,'random')
    # for metrics in metrics_list:
    #     result:  DefaultDict[str, DefaultDict[str, DefaultDict[str,List[float]]]] = defaultdict(lambda: defaultdict(lambda: dict))
    #     for model in model_list:
    #         for dataset in dataset_list:
    #             file = './result/{}/{}/{}/data/{}_{}_random.npy'.format(
    #                 metrics, model,dataset,control_factor, x_factor)
    #             item = np.load(file, allow_pickle=True)
    #             item = get_result(item, x_factor)
    #             result[model][dataset] = item
    #     draw_picture(result,control_factor,x_factor,'metrics',metrics,'random')

    for model in model_list:
        result:  DefaultDict[str, DefaultDict[str, DefaultDict[str, List[float]]]] = defaultdict(
            lambda: defaultdict(lambda: dict))

        control_factor = control_factor_list[0]
        for dataset in dataset_list:
            for metrics in metrics_list:
                file = './result/{}/{}/{}/data/{}_{}_random.npy'.format(
                    metrics, model, dataset, factor_to_file_dict[control_factor], factor_to_file_dict[x_factor])
                item = np.load(file, allow_pickle=True)
                item = get_result(item, control_factor_list, x_factor, metrics)
                result[dataset][metrics] = item
        draw_picture(result, control_factor_list, x_factor,
                     'model', model, 'random')


def main():
    random.seed(42)
    # metrics_list = ['rouge_1_f_score', 'rouge_2_f_score', 'rouge_3_f_score','rouge_4_f_score', 'rouge_l_f_score', 'rouge_su*_f_score', "rouge_w_1.2_f_score",  'rouge_we_3_f', 'bert_score_f1', 'mover_score', 'blanc', 'supert', 'bleu', 'chrf', 'meteor', 'summary_length', 'percentage_novel_1-gram','percentage_novel_2-gram', 'percentage_novel_3-gram','percentage_repeated_1-gram_in_summ', 'percentage_repeated_2-gram_in_summ','percentage_repeated_3-gram_in_summ', 'coverage', 'compression', 'density']
    metrics_list = ['rouge_1_f_score', 'rouge_2_f_score', 'rouge_3_f_score', 'rouge_4_f_score', 'rouge_l_f_score', 'rouge_su*_f_score',
                    "rouge_w_1.2_f_score",  'rouge_we_3_f', 'bert_score_f1', 'mover_score', 'blanc', 'bleu', 'chrf', 'meteor']
    model_list = ['bart-base', 'led-base-16384', 't5-base']
    # model_list = ['bart-base']
    dataset_list = ['cnn_dailymail', 'wikihowAll', 'corpus-webis-tldr-17']
    control_factor_list = ['article', 'reference']
    x_factor = 'generated summary'
    draw_result_picture(metrics_list, model_list,
                        dataset_list, control_factor_list, x_factor)
    draw_random_picture(metrics_list, model_list,
                        dataset_list, control_factor_list, x_factor)
    control_factor_list = ['generated summary']
    for x_factor in ['article', 'reference']:
        draw_result_picture(metrics_list, model_list,
                            dataset_list, control_factor_list, x_factor)
        draw_random_picture(metrics_list, model_list,
                            dataset_list, control_factor_list, x_factor)


if __name__ == '__main__':
    main()
