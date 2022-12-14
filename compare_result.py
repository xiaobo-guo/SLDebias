import os
from typing import Dict, List, Tuple, Union
import pandas as pd
import sys
import random
from nltk import tokenize
import numpy as np
import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import matplotlib.pyplot as plt
from bert_score import BERTScorer
import transformers
import logging
from copy import deepcopy


from summ_eval.rouge_metric import RougeMetric
from summ_eval.rouge_we_metric import RougeWeMetric
from summ_eval.mover_score_metric import MoverScoreMetric
from summ_eval.blanc_metric import BlancMetric
from summ_eval.supert_metric import SupertMetric
from summ_eval.blanc_metric import BlancMetric
from summ_eval.meteor_metric import MeteorMetric
from summ_eval.data_stats_metric import DataStatsMetric
from summ_eval.chrfpp_metric import ChrfppMetric

os.environ["TOKENIZERS_PARALLELISM"] = "false"
log_level = logging.ERROR
transformers.utils.logging.set_verbosity(log_level)


metrics_list = ['rouge-1', 'rouge-2', 'rouge-l', 'bleu', 'bertscore']
# metrics_list = ['rouge-1','rouge-2','rouge-l','bleu']
factor_list = ['article', 'generation', 'groundtruth']


def get_groundtruth(file_path: str) -> List[List[str]]:
    data = list()
    raw_data = pd.read_csv(file_path)
    raw_data = raw_data.values.tolist()
    for i, item in enumerate(raw_data):
        if len(item) == 2:
            item = [i] + item
        if len(item[1].split(' ')) < 10 or len(item[2].split(' ')) < 10:
            continue
        data.append(item)
    return data


def get_summarization(file_path: str) -> List[str]:
    data = list()
    with open(file_path, mode='r', encoding='utf8') as fp:
        for line in fp.readlines():
            data.append('\n'.join(tokenize.sent_tokenize(line.strip())))
    return data


def combine_data(groundtruth_data: List[List[Union[str, int, List[str]]]], generated_data: List[str]) -> Tuple[List[Dict[str, Union[str, int]]], Dict[str, List[int]]]:
    data = list()
    error_count = 0
    for i, item in enumerate(groundtruth_data):
        if len(item) == 3:
            item.append(len(item[1].replace('\n', ' ').split(' ')))
            item.append(len(item[2].replace('\n', ' ').split(' ')))
            item.append(list())
            item.append(list())
        item[5].append(len(generated_data[i].replace('\n', ' ').split(' ')))
        item[6].append(generated_data[i])
        data.append(item)
    return data


def calcualte_length(groundtruth_data: List[List[str]], data) -> Dict[str, List[int]]:
    statistics_data = {'article_lens': list(
    ), 'groundtruth_lens': list(), 'generation_lens': list()}
    for i, item in enumerate(groundtruth_data):
        item_data = {'id': item[0], 'article': item[1], 'groundtruth': item[2]}
        item_data['article_lens'] = len(
            item_data['article'].replace('\n', ' ').split(' '))
        item_data['groundtruth_lens'] = len(
            item_data['groundtruth'].replace('\n', ' ').split(' '))
        item_data['generation_lens'] = np.mean(data[i][5])
        data[i][5] = np.mean(data[i][5])

        statistics_data['article_lens'].append(item_data['article_lens'])
        statistics_data['groundtruth_lens'].append(
            item_data['groundtruth_lens'])
        statistics_data['generation_lens'].append(item_data['generation_lens'])
    return statistics_data


def calculate_score_splited(data, mode):
    score_list = dict()

    if mode == 'all':
        mode = ['cpu', 'gpu']
    else:
        mode = [mode]

    if 'cpu' in mode:
        temp_score_list = calculate_score_splited_cpu(data)
        for k, v in temp_score_list.items():
            score_list[k] = v

    if 'gpu' in mode:
        temp_score_list = calculate_score_splited_gpu(data)
        for k, v in temp_score_list.items():
            score_list[k] = v

    return score_list


def calculate_score_splited_cpu(data):
    total_item = 0
    for it in data:
        total_item += len(it)
    total_item = total_item

    chencherry = SmoothingFunction()

    rouge_model = RougeMetric()
    rougewe_model = RougeWeMetric()
    meteor_model = MeteorMetric()
    datastats_model = DataStatsMetric()
    chrfpp_model = ChrfppMetric()

    score_list = dict()
    for splited_data in tqdm.tqdm(data, desc='calculate score', ncols=100):
        splited_score_list = dict()

        article_list = [item[1].replace('\n', ' ') for item in splited_data]
        reference_list = [item[2].replace('\n', ' ') for item in splited_data]
        summary_list = [item[-1][0].replace('\n', ' ')
                        for item in splited_data]

        article_lens_list = [item[3] for item in splited_data]
        reference_lens_list = [item[4] for item in splited_data]
        summary_lens_list = [item[5] for item in splited_data]

        score_dict = rouge_model.evaluate_batch(
            summaries=summary_list, references=reference_list, aggregate=False)
        for i, score in enumerate(score_dict):
            try:
                for k, v in score['rouge'].items():
                    if k.endswith('_f_score'):
                        if k not in splited_score_list:
                            splited_score_list[k] = list()
                        splited_score_list[k].append(
                            (article_lens_list[i], reference_lens_list[i], summary_lens_list[i], v*100))
            except:
                for k, v in score.items():
                    if k.endswith('_f_score'):
                        if k not in splited_score_list:
                            splited_score_list[k] = list()
                        splited_score_list[k].append(
                            (article_lens_list[i], reference_lens_list[i], summary_lens_list[i], v*100))

        score_dict = rougewe_model.evaluate_batch(
            summaries=summary_list, references=reference_list, aggregate=False)
        for i, score in enumerate(score_dict):
            for k, v in score.items():
                if k.endswith('_f'):
                    if k not in splited_score_list:
                        splited_score_list[k] = list()
                    splited_score_list[k].append(
                        (article_lens_list[i], reference_lens_list[i], summary_lens_list[i], v*100))

        score_dict = meteor_model.evaluate_batch(
            summaries=summary_list, references=reference_list, aggregate=False)
        for i, score in enumerate(score_dict):
            for k, v in score.items():
                if k not in splited_score_list:
                    splited_score_list[k] = list()
                splited_score_list[k].append(
                    (article_lens_list[i], reference_lens_list[i], summary_lens_list[i], v*100))

        score_dict = datastats_model.evaluate_batch(
            summaries=summary_list, input_texts=article_list, aggregate=False)
        for i, score in enumerate(score_dict):
            for k, v in score.items():
                if k not in splited_score_list:
                    splited_score_list[k] = list()
                splited_score_list[k].append(
                    (article_lens_list[i], reference_lens_list[i], summary_lens_list[i], v*100))

        score_dict = chrfpp_model.evaluate_batch(
            summaries=summary_list, references=reference_list, aggregate=False)
        for i, score in enumerate(score_dict):
            for k, v in score.items():
                if k not in splited_score_list:
                    splited_score_list[k] = list()
                splited_score_list[k].append(
                    (article_lens_list[i], reference_lens_list[i], summary_lens_list[i], v*100))

        for i, item in enumerate(splited_data):
            groundtruth = item[2]
            summary = item[-1][0]
            bleu = sentence_bleu(references=[groundtruth.replace('\n', ' ').split(
            )], hypothesis=summary.replace('\n', ' ').split(), smoothing_function=chencherry.method1)
            if 'bleu' not in splited_score_list:
                splited_score_list['bleu'] = list()
            splited_score_list['bleu'].append(
                (article_lens_list[i], reference_lens_list[i], summary_lens_list[i], bleu*100))

        for metrics, v in splited_score_list.items():
            if metrics not in score_list:
                score_list[metrics] = list()
            score_list[metrics].append(v)
    return score_list


def calculate_score_splited_gpu(data):
    total_item = 0
    for it in data:
        total_item += len(it)
    total_item = total_item

    moverscore_model = MoverScoreMetric()
    Bertscore_model = BERTScorer(lang="en", rescale_with_baseline=True)
    blanc_model = BlancMetric(inference_batch_size=64, finetune_batch_size=12)
    supert_model = SupertMetric()

    score_list = dict()
    for splited_data in tqdm.tqdm(data, desc='calculate score', ncols=100):
        splited_score_list = dict()

        article_list = [item[1].replace('\n', ' ') for item in splited_data]
        reference_list = [item[2].replace('\n', ' ') for item in splited_data]
        summary_list = [item[-1][0].replace('\n', ' ')
                        for item in splited_data]

        article_lens_list = [item[3] for item in splited_data]
        reference_lens_list = [item[4] for item in splited_data]
        summary_lens_list = [item[5] for item in splited_data]

        score_dict = moverscore_model.evaluate_batch(
            summaries=summary_list, references=reference_list, aggregate=False)
        for i, score in enumerate(score_dict):
            for k, v in score.items():
                if k not in splited_score_list:
                    splited_score_list[k] = list()
                splited_score_list[k].append(
                    (article_lens_list[i], reference_lens_list[i], summary_lens_list[i], v*100))

        _, _, bert_score = Bertscore_model.score(
            cands=summary_list, refs=reference_list, batch_size=12)
        if 'bert_score_f1' not in splited_score_list:
            splited_score_list['bert_score_f1'] = list()
        for i, score in enumerate(bert_score.numpy().tolist()):
            splited_score_list['bert_score_f1'].append(
                (article_lens_list[i], reference_lens_list[i], summary_lens_list[i], score*100))

        # score_dict = summaqa_model.evaluate_batch(summaries = summary_list, input_texts = article_list, aggregate=False)
        # for i, score in enumerate(score_dict):
        #     for k,v in score.items():
        #         if k not in splited_score_list:
        #             splited_score_list[k] = list()
        #         splited_score_list[k].append((article_lens_list[i], reference_lens_list[i], summary_lens_list[i], v*100 ))

        score_dict = blanc_model.evaluate_batch(
            summaries=summary_list, input_texts=article_list, aggregate=False, show_progress_bar=False)
        for i, score in enumerate(score_dict):
            for k, v in score.items():
                if k not in splited_score_list:
                    splited_score_list[k] = list()
                splited_score_list[k].append(
                    (article_lens_list[i], reference_lens_list[i], summary_lens_list[i], v*100))

        score_dict = supert_model.evaluate_batch(
            summaries=summary_list, input_texts=article_list, aggregate=False)
        for i, score in enumerate(score_dict):
            for k, v in score.items():
                if k not in splited_score_list:
                    splited_score_list[k] = list()
                try:
                    splited_score_list[k].append(
                        (article_lens_list[i], reference_lens_list[i], summary_lens_list[i], v*100))
                except:
                    splited_score_list[k].append(
                        (article_lens_list[i], reference_lens_list[i], summary_lens_list[i], 0))

        for metrics, v in splited_score_list.items():
            if metrics not in score_list:
                score_list[metrics] = list()
            score_list[metrics].append(v)

    return score_list


def calculate_score(data, model_name, mode):
    score_list = dict()

    if mode == 'all':
        mode = ['cpu', 'gpu']
    else:
        mode = [mode]

    if 'cpu' in mode:
        temp_score_list = calculate_score_cpu(data, model_name)
        for k, v in temp_score_list.items():
            score_list[k] = v

    if 'gpu' in mode:
        temp_score_list = calculate_score_gpu(data, model_name)
        for k, v in temp_score_list.items():
            score_list[k] = v

    return score_list


def calculate_score_cpu(data, model_name):
    score_list = dict()

    article_list = [item[1].replace('\n', ' ') for item in data]
    reference_list = [item[2].replace('\n', ' ') for item in data]
    summary_list = [item[-1][0].replace('\n', ' ') for item in data]

    article_lens_list = [item[3] for item in data]
    reference_lens_list = [item[4] for item in data]
    summary_lens_list = [item[5] for item in data]

    datastats_model = DataStatsMetric()

    print('datasets Start')
    score_dict = datastats_model.evaluate_batch(
        summaries=summary_list, input_texts=article_list, aggregate=False, show_progress_bar=True)

    if model_name == 'groundtruth':
        for i, score in enumerate(score_dict):
            for k, v in score.items():
                if k not in score_list:
                    score_list[k] = list()
                score_list[k].append(
                    (article_lens_list[i], reference_lens_list[i], summary_lens_list[i], v*100))

        return score_list

    chencherry = SmoothingFunction()
    rouge_model = RougeMetric()
    rougewe_model = RougeWeMetric()
    meteor_model = MeteorMetric()
    chrfpp_model = ChrfppMetric()

    print('Rouge Start')
    score_dict = rouge_model.evaluate_batch(
        summaries=summary_list, references=reference_list, aggregate=False, show_progress_bar=True)
    for i, score in enumerate(score_dict):
        try:
            for k, v in score['rouge'].items():
                if k.endswith('_f_score'):
                    if k not in score_list:
                        score_list[k] = list()
                    score_list[k].append(
                        (article_lens_list[i], reference_lens_list[i], summary_lens_list[i], v*100))
        except:
            for k, v in score.items():
                if k.endswith('_f_score'):
                    if k not in score_list:
                        score_list[k] = list()
                    score_list[k].append(
                        (article_lens_list[i], reference_lens_list[i], summary_lens_list[i], v*100))

    print('RougeWe Start')
    score_dict = rougewe_model.evaluate_batch(
        summaries=summary_list, references=reference_list, aggregate=False, show_progress_bar=True)
    for i, score in enumerate(score_dict):
        for k, v in score.items():
            if k.endswith('_f'):
                if k not in score_list:
                    score_list[k] = list()
                score_list[k].append(
                    (article_lens_list[i], reference_lens_list[i], summary_lens_list[i], v*100))

    print('meteor Start')
    score_dict = meteor_model.evaluate_batch(
        summaries=summary_list, references=reference_list, aggregate=False, show_progress_bar=True)
    for i, score in enumerate(score_dict):
        for k, v in score.items():
            if k not in score_list:
                score_list[k] = list()
            score_list[k].append(
                (article_lens_list[i], reference_lens_list[i], summary_lens_list[i], v*100))

    print("chrfpp starts")
    score_dict = chrfpp_model.evaluate_batch(
        summaries=summary_list, references=reference_list, aggregate=False, show_progress_bar=True)
    for i, score in enumerate(score_dict):
        for k, v in score.items():
            if k not in score_list:
                score_list[k] = list()
            score_list[k].append(
                (article_lens_list[i], reference_lens_list[i], summary_lens_list[i], v*100))

    print('BLEU Start')
    for _, item in enumerate(tqdm.tqdm(data, ncols=True, desc='Calculate BLEU')):
        groundtruth = item[2]
        summary = item[-1][0]
        bleu = sentence_bleu(references=[groundtruth.replace('\n', ' ').split(
        )], hypothesis=summary.replace('\n', ' ').split(), smoothing_function=chencherry.method1)
        if 'bleu' not in score_list:
            score_list['bleu'] = list()
        score_list['bleu'].append((item[3], item[4], item[5], bleu*100))

    for i, score in enumerate(score_dict):
        for k, v in score.items():
            if k not in score_list:
                score_list[k] = list()
            score_list[k].append(
                (article_lens_list[i], reference_lens_list[i], summary_lens_list[i], v*100))

    return score_list


def calculate_score_gpu(data, model_name):
    score_list = dict()

    article_list = [item[1].replace('\n', ' ') for item in data]
    reference_list = [item[2].replace('\n', ' ') for item in data]
    summary_list = [item[-1][0].replace('\n', ' ') for item in data]

    article_lens_list = [item[3] for item in data]
    reference_lens_list = [item[4] for item in data]
    summary_lens_list = [item[5] for item in data]

    # summaqa_model =  SummaQAMetric(max_seq_len=4096, batch_size=4)
    blanc_model = BlancMetric(inference_batch_size=64, finetune_batch_size=12)
    supert_model = SupertMetric()

    print('Supert Start')
    score_dict = supert_model.evaluate_batch(
        summaries=summary_list, input_texts=article_list, aggregate=False, show_progress_bar=True)
    for i, score in enumerate(score_dict):
        for k, v in score.items():
            if k not in score_list:
                score_list[k] = list()
            try:
                score_list[k].append(
                    (article_lens_list[i], reference_lens_list[i], summary_lens_list[i], v*100))
            except:
                score_list[k].append(
                    (article_lens_list[i], reference_lens_list[i], summary_lens_list[i], 0))

    # print('Summaqa Start')
    # score_dict = summaqa_model.evaluate_batch(summaries = summary_list, input_texts = article_list, aggregate=False)
    # for i, score in enumerate(score_dict):
    #     for k,v in score.items():
    #         if k not in score_list:
    #             score_list[k] = list()
    #         score_list[k].append((article_lens_list[i], reference_lens_list[i], summary_lens_list[i], v*100 ))

    print('Blanc Start')
    score_dict = blanc_model.evaluate_batch(
        summaries=summary_list, input_texts=article_list, aggregate=False)

    if model_name == 'groundtruth':
        for i, score in enumerate(score_dict):
            for k, v in score.items():
                if k not in score_list:
                    score_list[k] = list()
                score_list[k].append(
                    (article_lens_list[i], reference_lens_list[i], summary_lens_list[i], v*100))

        return score_list

    moverscore_model = MoverScoreMetric()
    Bertscore_model = BERTScorer(lang="en", rescale_with_baseline=True)
    # summaqa_model =  SummaQAMetric(max_seq_len=4096, batch_size=4)

    print('MoverScore Start')
    score_dict = moverscore_model.evaluate_batch(
        summaries=summary_list, references=reference_list, aggregate=False, show_progress_bar=True)
    for i, score in enumerate(score_dict):
        for k, v in score.items():
            if k not in score_list:
                score_list[k] = list()
            score_list[k].append(
                (article_lens_list[i], reference_lens_list[i], summary_lens_list[i], v*100))

    print('Bertscore Start')
    _, _, bert_score = Bertscore_model.score(
        cands=summary_list, refs=reference_list, batch_size=12)
    if 'bert_score_f1' not in score_list:
        score_list['bert_score_f1'] = list()
    for i, score in enumerate(bert_score.numpy().tolist()):
        score_list['bert_score_f1'].append(
            (article_lens_list[i], reference_lens_list[i], summary_lens_list[i], score*100))

    for i, score in enumerate(score_dict):
        for k, v in score.items():
            if k not in score_list:
                score_list[k] = list()
            score_list[k].append(
                (article_lens_list[i], reference_lens_list[i], summary_lens_list[i], v*100))

    return score_list


def split_data_shuffle(statistics_data, data, split_base):
    split_number = 10
    if split_base == 'article':
        split_index = 3
    elif split_base == 'groundtruth':
        split_index = 4
    elif split_base == 'generation':
        split_index = 5

    random.shuffle(data)
    sorted_shuffled_item = sorted(data, key=lambda x: x[split_index])
    split_count = int(len(sorted_shuffled_item) / split_number)
    bucket_data = [[] for _ in range(split_number)]
    for j in range(split_number):
        bucket_data[j] = [
            it for it in sorted_shuffled_item[j*split_count:(j+1)*split_count]]
    if split_count * split_number != len(data):
        for it in sorted_shuffled_item[split_number*split_count:]:
            bucket_data[j].append(it)
    return bucket_data


def split_data(statistics_data, data, split_base):
    splited_data = dict()
    for k, score_data in data.items():
        split_number = 10
        split_index = 0
        if split_base == 'article':
            split_index = 0
        elif split_base == 'groundtruth':
            split_index = 1
        elif split_base == 'generation':
            split_index = 2

        random.shuffle(score_data)
        sorted_shuffled_item = sorted(score_data, key=lambda x: x[split_index])
        split_count = int(len(sorted_shuffled_item) / split_number)
        bucket_data = [[] for _ in range(split_number)]
        for j in range(split_number):
            bucket_data[j] = [
                it for it in sorted_shuffled_item[j*split_count:(j+1)*split_count]]
        if split_count * split_number != len(score_data):
            for it in sorted_shuffled_item[split_number*split_count:]:
                bucket_data[j].append(it)
        splited_data[k] = bucket_data
    return splited_data


def shuffle_data(data):
    shuffled_data = list()
    for splited_data in data:
        generation_list = list()
        for item in splited_data:
            generation_list.append((item[5], item[-1]))
        old_generation_list = deepcopy(generation_list)
        difference_mark = False
        while not difference_mark:
            check_mark = True
            random.shuffle(generation_list)
            for old_generation, generation in zip(old_generation_list, generation_list):
                if old_generation == generation:
                    check_mark = False
                    break
            if check_mark:
                difference_mark = True
        for i, item in enumerate(splited_data):
            item[-1] = generation_list[i][1]
            item[5] = generation_list[i][0]
        shuffled_data.append(splited_data)
    return shuffled_data


def filter_data(groundtruth_data, data):
    filtered_groundtruth_data = list()
    filtered_data = list()
    for i, item in enumerate(data):
        if item[3] < 10 or item[4] < 10 or np.mean(item[5]) < 10:
            continue
        filtered_groundtruth_data.append(groundtruth_data[i])
        filtered_data.append(data[i])
    return filtered_groundtruth_data, filtered_data


def save_score(bucket_data, control_factor, split_base, metrics, model_name, dataset_name, random=False):
    bucket_data_np = np.array(bucket_data, dtype=object)
    model_name = model_name.split('/')[-1]
    folder = "./result/{}/{}/{}/data".format(metrics, model_name, dataset_name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    file = "./result/{}/{}/{}/data/{}_{}".format(
        metrics, model_name, dataset_name, control_factor, split_base)
    if random == True:
        file = file + '_random'
    np.save(file, bucket_data_np)


def print_reslut(bucket_data, control_factor, split_base, metrics, model_name, dataset_name, random_shuffle=False):
    model_name = model_name.split('/')[-1]
    split_number = 10
    split_index = 0
    if split_base == 'article':
        split_index = 0
    elif split_base == 'groundtruth':
        split_index = 1
    elif split_base == 'generation':
        split_index = 2

    for i, item in enumerate(bucket_data):
        random.shuffle(item)
        sorted_shuffled_item = sorted(item, key=lambda x: x[split_index])
        split_count = int(len(sorted_shuffled_item) / split_number)
        bucket = [[] for _ in range(split_number)]
        for j in range(split_number):
            bucket[j] = [it[-1]
                         for it in sorted_shuffled_item[j*split_count:(j+1)*split_count]]
        if split_count * split_number != len(item):
            for it in sorted_shuffled_item[split_number*split_count:]:
                bucket[j].append(it[-1])
        y = [np.mean(d) for d in bucket]
        plt.plot([1/split_number*j for j in range(split_number)], y,
                 label='{}% length of {}'.format((i+1)*len(bucket_data), control_factor))
    plt.title("The performance when controlling {} length".format(control_factor))
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.xlabel("percent of length of {}".format(split_base))
    folder = "./result/{}/{}/{}/picture".format(
        metrics, model_name, dataset_name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    file = "./result/{}/{}/{}/picture/{}_{}.jpg".format(
        metrics, model_name, dataset_name, control_factor, split_base)
    if random_shuffle == True:
        file = file.replace('.jpg', '_random.jpg')
    plt.savefig(file, bbox_inches='tight')

    plt.close()


def main():
    # shuffle=False

    random.seed(42)
    # dataset_name = ''
    # for model_name in ['facebook/bart-base', 't5-base', 'allenai/led-base-16384']:

    dataset_name = sys.argv[1]
    model_name = sys.argv[2]
    mode = sys.argv[3]

    # dataset_name = 'wikihowAll'
    # model_name = 'groundtruth'
    # mode = 'cpu'

    groudtruth_file = './data/{}/test.csv'.format(dataset_name)
    groundtruth_data = get_groundtruth(groudtruth_file)
    if model_name == 'groundtruth':
        summary_data = [item[2] for item in groundtruth_data]
        data = combine_data(groundtruth_data, summary_data)
    else:
        for random_seed in [0]:
            generated_file = './model/{}/{}/{}/generated_predictions.txt'.format(
                dataset_name, model_name, random_seed)
            summary_data = get_summarization(generated_file)
            data = combine_data(groundtruth_data, summary_data)
        if dataset_name == 'corpus-webis-tldr-17':
            groundtruth_data, data = filter_data(groundtruth_data, data)
    statistics_data = calcualte_length(groundtruth_data, data)

    # score = calculate_score(data, model_name, mode)
    # for control_factor in factor_list:
    #     splited_score = split_data(statistics_data, score, control_factor)
    #     for x_factor in factor_list:
    #         if x_factor == control_factor:
    #             continue
    #         for metrics in tqdm.tqdm(list(splited_score.keys()), desc='create picture', ncols=100):
    #             save_score(splited_score[metrics], control_factor,
    #                        x_factor, metrics, model_name, dataset_name)
    #             print_reslut(splited_score[metrics], control_factor,
    #                          x_factor, metrics, model_name, dataset_name)

    for control_factor in factor_list:
        splited_data = split_data_shuffle(
            statistics_data, data, control_factor)
        splited_data = shuffle_data(splited_data)
        splited_score = calculate_score_splited(splited_data, mode)
        for x_factor in factor_list:
            if x_factor == control_factor:
                continue
            for metrics in tqdm.tqdm(list(splited_score.keys()), desc='create picture', ncols=100):
                save_score(splited_score[metrics], control_factor,
                           x_factor, metrics, model_name, dataset_name, True)
                print_reslut(splited_score[metrics], control_factor,
                             x_factor, metrics, model_name, dataset_name, True)


if __name__ == '__main__':
    main()
