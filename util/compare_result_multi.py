from summ_eval.chrfpp_metric import ChrfppMetric
from summ_eval.cider_metric import CiderMetric
from summ_eval.syntactic_metric import SyntacticMetric
from summ_eval.data_stats_metric import DataStatsMetric
from summ_eval.s3_metric import S3Metric
from summ_eval.meteor_metric import MeteorMetric
from summ_eval.supert_metric import SupertMetric
from summ_eval.blanc_metric import BlancMetric
from summ_eval.summa_qa_metric import SummaQAMetric
from summ_eval.sentence_movers_metric import SentenceMoversMetric
from summ_eval.mover_score_metric import MoverScoreMetric
from summ_eval.rouge_we_metric import RougeWeMetric
from summ_eval.rouge_metric import RougeMetric
from enum import EnumMeta
from importlib import import_module
import os
from re import L
from typing import Dict, List, Tuple, Union
import pandas as pd
import sys
import random
from nltk import tokenize
import numpy as np
import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu
import matplotlib.pyplot as plt
from bert_score import BERTScorer
import transformers
import logging
import json
import warnings
warnings.simplefilter("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
log_level = logging.ERROR
transformers.utils.logging.set_verbosity(log_level)


metrics_list = ['rouge-1', 'rouge-2', 'rouge-l', 'bleu', 'bertscore']
# metrics_list = ['rouge-1','rouge-2','rouge-l','bleu']
factor_list = ['article', 'groundtruth', 'generation']


def get_groundtruth(file_path: str) -> List[List[str]]:
    data = pd.read_csv(file_path)
    data = data.values.tolist()
    return data


def get_data(file_path: str) -> List[str]:
    data = list()
    with open(file_path, mode='r', encoding='utf8') as fp:
        for line in fp.readlines():
            item = json.loads(line.strip())
            data.append(item)
    return data


def combine_data(groundtruth_data: List[List[Union[str, int, List[str]]]], generated_data: List[str]) -> Tuple[List[Dict[str, Union[str, int]]], Dict[str, List[int]]]:
    data = list()
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


def calcualte_length(data: List[Dict[str, str]]) -> Dict[str, List[int]]:
    statistics_data = {'article_lens': list(
    ), 'groundtruth_lens': list(), 'generation_lens': list()}
    for i, item in enumerate(data):
        item_data = dict()
        item_data['article_lens'] = len(
            item['text'].replace('\n', ' ').split(' '))
        item_data['groundtruth_lens'] = len(
            item['reference'].replace('\n', ' ').split(' '))
        item_data['generation_lens'] = len(
            item['decoded'].replace('\n', ' ').split(' '))

        item['generation_lens'] = item_data['generation_lens']
        item['article_lens'] = item_data['article_lens']
        item['groundtruth_lens'] = item_data['groundtruth_lens']

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

        article_list = [item['text'].replace(
            '\n', ' ') for item in splited_data]
        reference_list = [item['reference'].replace(
            '\n', ' ') for item in splited_data]
        summary_list = [item['decoded'].replace(
            '\n', ' ') for item in splited_data]

        article_lens_list = [item['article_lens'] for item in splited_data]
        reference_lens_list = [item['groundtruth_lens']
                               for item in splited_data]
        summary_lens_list = [item['generation_lens'] for item in splited_data]

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
            groundtruth = item['reference']
            summary = item['decoded']
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

        article_list = [item['text'].replace(
            '\n', ' ') for item in splited_data]
        reference_list = [item['reference'].replace(
            '\n', ' ') for item in splited_data]
        summary_list = [item['decoded'].replace(
            '\n', ' ') for item in splited_data]

        article_lens_list = [item['article_lens'] for item in splited_data]
        reference_lens_list = [item['groundtruth_lens']
                               for item in splited_data]
        summary_lens_list = [item['generation_lens'] for item in splited_data]

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


def calculate_score(data, mode):
    score_list = dict()

    if mode == 'all':
        mode = ['cpu', 'gpu']
    else:
        mode = [mode]

    if 'cpu' in mode:
        temp_score_list = calculate_score_cpu(data)
        for k, v in temp_score_list.items():
            score_list[k] = v

    if 'gpu' in mode:
        temp_score_list = calculate_score_gpu(data)
        for k, v in temp_score_list.items():
            score_list[k] = v

    return score_list


def calculate_score_cpu(data):
    score_list = dict()

    article_list = [item['text'].replace('\n', ' ') for item in data]
    reference_list = [item['reference'].replace('\n', ' ') for item in data]
    summary_list = [item['decoded'].replace('\n', ' ') for item in data]

    article_lens_list = [item['article_lens'] for item in data]
    reference_lens_list = [item['groundtruth_lens'] for item in data]
    summary_lens_list = [item['generation_lens'] for item in data]

    chencherry = SmoothingFunction()
    rouge_model = RougeMetric()
    rougewe_model = RougeWeMetric()
    meteor_model = MeteorMetric()
    datastats_model = DataStatsMetric()
    chrfpp_model = ChrfppMetric()

    print('Rouge Start')
    score_dict = rouge_model.evaluate_batch(
        summaries=summary_list, references=reference_list, aggregate=False)
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
        summaries=summary_list, references=reference_list, aggregate=False)
    for i, score in enumerate(score_dict):
        for k, v in score.items():
            if k.endswith('_f'):
                if k not in score_list:
                    score_list[k] = list()
                score_list[k].append(
                    (article_lens_list[i], reference_lens_list[i], summary_lens_list[i], v*100))

    print('meteor Start')
    score_dict = meteor_model.evaluate_batch(
        summaries=summary_list, references=reference_list, aggregate=False)
    for i, score in enumerate(score_dict):
        for k, v in score.items():
            if k not in score_list:
                score_list[k] = list()
            score_list[k].append(
                (article_lens_list[i], reference_lens_list[i], summary_lens_list[i], v*100))

    print('datasets Start')
    score_dict = datastats_model.evaluate_batch(
        summaries=summary_list, input_texts=article_list, aggregate=False)
    for i, score in enumerate(score_dict):
        for k, v in score.items():
            if k not in score_list:
                score_list[k] = list()
            score_list[k].append(
                (article_lens_list[i], reference_lens_list[i], summary_lens_list[i], v*100))

    print("chrfpp starts")
    score_dict = chrfpp_model.evaluate_batch(
        summaries=summary_list, references=reference_list, aggregate=False)
    for i, score in enumerate(score_dict):
        for k, v in score.items():
            if k not in score_list:
                score_list[k] = list()
            score_list[k].append(
                (article_lens_list[i], reference_lens_list[i], summary_lens_list[i], v*100))

    print('BLEU Start')
    for i, item in enumerate(data):
        groundtruth = item['reference']
        summary = item['decoded']
        bleu = sentence_bleu(references=[groundtruth.replace('\n', ' ').split(
        )], hypothesis=summary.replace('\n', ' ').split(), smoothing_function=chencherry.method1)
        if 'bleu' not in score_list:
            score_list['bleu'] = list()
        score_list['bleu'].append(
            (article_lens_list[i], reference_lens_list[i], summary_lens_list[i], bleu*100))

    return score_list


def calculate_score_gpu(data):
    score_list = dict()
    article_list = [item['text'].replace('\n', ' ') for item in data]
    reference_list = [item['reference'].replace('\n', ' ') for item in data]
    summary_list = [item['decoded'].replace('\n', ' ') for item in data]

    article_lens_list = [item['article_lens'] for item in data]
    reference_lens_list = [item['groundtruth_lens'] for item in data]
    summary_lens_list = [item['generation_lens'] for item in data]

    moverscore_model = MoverScoreMetric()
    Bertscore_model = BERTScorer(lang="en", rescale_with_baseline=True)
    # summaqa_model =  SummaQAMetric(max_seq_len=4096, batch_size=4)
    blanc_model = BlancMetric(inference_batch_size=64, finetune_batch_size=12)
    supert_model = SupertMetric()

    print('Blanc Start')
    score_dict = blanc_model.evaluate_batch(
        summaries=summary_list, input_texts=article_list, aggregate=False)
    for i, score in enumerate(score_dict):
        for k, v in score.items():
            if k not in score_list:
                score_list[k] = list()
            score_list[k].append(
                (article_lens_list[i], reference_lens_list[i], summary_lens_list[i], v*100))

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

    # print('Summaqa Start')
    # score_dict = summaqa_model.evaluate_batch(summaries = summary_list, input_texts = article_list, aggregate=False)
    # for i, score in enumerate(score_dict):
    #     for k,v in score.items():
    #         if k not in score_list:
    #             score_list[k] = list()
    #         score_list[k].append((article_lens_list[i], reference_lens_list[i], summary_lens_list[i], v*100 ))

    print('Supert Start')
    score_dict = supert_model.evaluate_batch(
        summaries=summary_list, input_texts=article_list, aggregate=False)
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
    return score_list


def split_data(statistics_data, data, split_base):
    splited_data = dict()
    split_percent = 10
    length_split = statistics_data['{}_lens'.format(split_base)]
    length_split = sorted(length_split)
    split_number_list = [int(i*(1/split_percent)*len(length_split)-1)
                         for i in range(1, split_percent+1)]
    split_number_list = [length_split[i] for i in split_number_list]
    index = 0
    if split_base == 'article':
        index = 0
    elif split_base == 'groundtruth':
        index = 1
    elif split_base == 'generation':
        index = 2

    for k, score_data in data.items():
        bucket_data = [[] for _ in range(split_percent)]
        for item in score_data:
            for i in range(split_percent):
                if item[index] < split_number_list[i]:
                    bucket_data[i].append(item)
                    break
        splited_data[k] = bucket_data
    return splited_data


def split_data_shuffle(statistics_data, data, split_base):
    split_percent = 10

    length_split = statistics_data['{}_lens'.format(split_base)]
    length_split = sorted(length_split)
    split_number_list = [int(i*(1/split_percent)*len(length_split)-1)
                         for i in range(1, split_percent+1)]
    split_number_list = [length_split[i] for i in split_number_list]
    bucket_data = [[] for _ in range(split_percent)]
    for item in data:
        for i in range(split_percent):
            if item['{}_lens'.format(split_base)] < split_number_list[i]:
                bucket_data[i].append(item)
                break
    return bucket_data


def shuffle_data(data):
    shuffled_data = list()
    for splited_data in data:
        generation_list = list()
        for item in splited_data:
            generation_list.append((item['reference'], item['decoded']))
        random.shuffle(generation_list)
        for i, item in enumerate(splited_data):
            item['decoded'] = generation_list[i][1]
            item['reference'] = generation_list[i][0]
        shuffled_data.append(splited_data)
    return shuffled_data


def print_reslut(bucket_data, control_factor, split_base, metrics, model_name, dataset_name, random=False):
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
        base_length = [it[split_index] for it in item]
        base_length_split = sorted(base_length)
        split_number_list = [int(i*1/split_number*len(base_length_split)-1)
                             for i in range(1, split_number+1)]
        split_number_list = [base_length_split[i] for i in split_number_list]

        bucket = [[] for _ in range(split_number)]
        for d in item:
            index = 0
            while d[split_index] > split_number_list[index]:
                index += 1
            bucket[index].append(d[-1])
        y = [np.mean(d) for d in bucket]
        plt.plot([1/split_number*j for j in range(split_number)], y,
                 label='{}% length of {}'.format((i+1)*len(bucket_data), control_factor))
    plt.title("The performance when controlling {} length".format(control_factor))
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.xlabel("percent of length of {}".format(split_base))
    folder = "./result_new/{}/{}/{}/picture".format(
        metrics, model_name, dataset_name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    file = "./result_new/{}/{}/{}/picture/{}_{}.jpg".format(
        metrics, model_name, dataset_name, control_factor, split_base)
    if random == True:
        file = file.replace('.jpg', '_random.jpg')
    plt.savefig(file, bbox_inches='tight')

    plt.close()


def save_score(bucket_data, control_factor, split_base, metrics, model_name, dataset_name, random=False):
    bucket_data_np = np.array(bucket_data, dtype=object)
    model_name = model_name.split('/')[-1]
    folder = "./result_new/{}/{}/{}/data".format(
        metrics, model_name, dataset_name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    file = "./result_new/{}/{}/{}/data/{}_{}".format(
        metrics, model_name, dataset_name, control_factor, split_base)
    if random == True:
        file = file + '_random'
    np.save(file, bucket_data_np)


def main():
    # shuffle=False

    random.seed(42)
    dataset_name = 'cnndm'
    model_name = sys.argv[1]
    mode = sys.argv[2]

    # model_name="M23"
    # mode='cpu'
    print(model_name)

    # generated_file = './data/{}/outputs/{}/paired/outputs.aligned.paired.jsonl'.format(dataset_name, model_name)
    generated_file_folder = './data/{}/outputs/{}/paired'.format(
        dataset_name, model_name)
    generated_file = os.path.join(
        generated_file_folder, os.listdir(generated_file_folder)[0])
    data = get_data(generated_file)
    statistics_data = calcualte_length(data)

    score = calculate_score(data, mode)
    for control_factor in factor_list:
        splited_score = split_data(statistics_data, score, control_factor)
        for x_factor in factor_list:
            if x_factor == control_factor:
                continue
            for metrics in tqdm.tqdm(list(splited_score.keys()), desc='create pictrue'):
                save_score(splited_score[metrics], control_factor,
                           x_factor, metrics, model_name, dataset_name)
                print_reslut(splited_score[metrics], control_factor,
                             x_factor, metrics, model_name, dataset_name)
    for control_factor in factor_list:
        splited_data = split_data_shuffle(
            statistics_data, data, control_factor)
        splited_data = shuffle_data(splited_data)
        splited_score = calculate_score_splited(splited_data, mode)
        for x_factor in factor_list:
            if x_factor == control_factor:
                continue
            for metrics in tqdm.tqdm(list(splited_score.keys()), desc='create pictrue'):
                save_score(splited_score[metrics], control_factor,
                           x_factor, metrics, model_name, dataset_name, True)
                print_reslut(splited_score[metrics], control_factor,
                             x_factor, metrics, model_name, dataset_name, True)

    # for shuffle in [False, True]:
    #     statistics_data = calcualte_length(data)
    #     for control_factor in factor_list:
    #         splited_data = split_data(statistics_data, data, control_factor)
    #         if shuffle:
    #             splited_data = shuffle_data(splited_data)
    #         score = calculate_score(splited_data)
    #         # score = calculate_score_back(splited_data)
    #         for x_factor in factor_list:
    #             if x_factor == control_factor:
    #                 continue
    #             for metrics in tqdm.tqdm(list(score.keys()),desc='create pictrue'):
    #                 print_reslut(score[metrics], control_factor, x_factor ,metrics, model_name, dataset_name, shuffle)


if __name__ == '__main__':
    main()
