import os
import sys
from typing import Dict, List, Tuple, Union, Set, DefaultDict
import pandas as pd
import random
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import transformers
import logging
import json
import argparse
from collections import defaultdict
import numpy as np
from argparse import Namespace
from dataclasses import dataclass, field
from scipy.stats import pearsonr, kendalltau
from pomegranate import BayesianNetwork, DiscreteDistribution, ConditionalProbabilityTable, Node
from multiprocessing import Pool
import sys
import copy
import json
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

os.environ["TOKENIZERS_PARALLELISM"] = "false"
log_level = logging.ERROR
transformers.utils.logging.set_verbosity(log_level)


@dataclass
class AnalyzeCorrelationInfo:
    """
    Arguments for analyize correaltion between model results and human annotation results
    """
    summ_ids: Set[str] = field(
        default_factory=set, metadata={
            "help": "The models used for calcualting the correlation"}
    )
    extractive_ids: List[str] = field(
        default_factory=list, metadata={
            "help": "The extractive models used for calcualting the correlation"}
    )
    abstractive_ids: List[str] = field(
        default_factory=list, metadata={
            "help": "The abstractive models used for calcualting the correlation"}
    )
    sorted_keys: List[str] = field(
        default_factory=list, metadata={
            "help": "The metrics used for calculating"}
    )
    table_names: List[str] = field(default_factory=list, metadata={
                                   "help": "The metrics name in the table used for calculating"})
    keys_to_normalized: Dict[str, str] = field(default_factory=dict, metadata={
                                               "help": "map between name of  the metric keys and the name of metrics used for normalization"})

    def __post_init__(self) -> None:
        # self.summ_ids = ['M0','M1','M2','M5','M8','M9','M10','M11','M12','M13','M14','M15','M17','M20','M22', 'M23']
        self.summ_ids = ['M1', 'M5', 'M8', 'M9', 'M10', 'M11',
                         'M12', 'M13', 'M14', 'M15', 'M17', 'M20', 'M22', 'M23']
        self.extractive_ids = ["M0", "M1", "M2", "M5"]
        self.abstractive_ids = list(
            set(self.summ_ids) - set(self.extractive_ids))
        self.sorted_keys = ['rouge_1_f_score', 'rouge_2_f_score', 'rouge_3_f_score', 'rouge_4_f_score', 'rouge_l_f_score', 'rouge_su*_f_score',
                            'rouge_w_1.2_f_score',  'rouge_we_3_f', 'bert_score_f1', 'mover_score', 'blanc', 'bleu', 'chrf', 'meteor']
        # self.sorted_keys=['rouge_1_f_score', 'rouge_2_f_score', 'rouge_3_f_score','rouge_4_f_score', 'rouge_l_f_score', 'rouge_su*_f_score', 'rouge_w_1.2_f_score',  'rouge_we_3_f', 'bert_score_f1', 'mover_score', 'blanc', 'supert', 'bleu', 'chrf', 'meteor', 'summary_length', 'percentage_novel_1-gram','percentage_novel_2-gram', 'percentage_novel_3-gram','percentage_repeated_1-gram_in_summ', 'percentage_repeated_2-gram_in_summ','percentage_repeated_3-gram_in_summ', 'coverage', 'compression', 'density']
        self.table_names = ['ROUGE-1 ', 'ROUGE-2 ', 'ROUGE-3  ', 'ROUGE-4 ', 'ROUGE-L  ', 'ROUGE-su* ', 'ROUGE-w  ', 'ROUGE-we-3  ', 'BertScore-f  ', 'MoverScore ', 'BLANC', 'BLEU  ', 'CHRF  ', 'METEOR  ', 'Length\\^  ',
                            'Novel unigram\\^ ', 'Novel bi-gram\\^ ', 'Novel tri-gram\\^  ', 'Repeated unigram\\^ ', 'Repeated bi-gram\\^ ', 'Repeated tri-gram\\^  ', 'Stats-coverage\\^ ', 'Stats-compression\\^ ', 'Stats-density\\^ ']


def get_params() -> Namespace:
    parser = argparse.ArgumentParser(description='Process argument')
    # parser.add_argument('--correlator_str', default='kendall',
    #                     help='which correlation metric to use')
    parser.add_argument('--debug', action='store_true',
                        help='only include extractive methods in calculations')
    parser.add_argument('--single_process', action='store_true',
                        help='only include extractive methods in calculations')
    parser.add_argument('--only_extractive', action='store_true',
                        help='only include extractive methods in calculations')
    parser.add_argument('--annotators_str', default='expert_annotations',
                        help='string specificying whether to use expert annotations or turker annotations')
    parser.add_argument('--only_abstractive', action='store_true',
                        help='only include abstractive methods in calculations')
    parser.add_argument('--subset', default=11,
                        help='how many references used to calculate metric scores for correlation calculations')
    parser.add_argument('--article_dir',
                        default="./data",
                        help="the dir for article files")
    parser.add_argument('--input_file',
                        default="./data/cnndm/human/model_annotations.aligned.scored.jsonl",
                        help="jsonl file with annotations and metric scores")
    parser.add_argument('--baseline', default='prediction',
                        help='the baseline for calculating')
    parser.add_argument('--split_type', default='percent',
                        help='the baseline for calculating')
    parser.add_argument('--split_size', type=int, default=10,
                        help='the baseline for calculating')
    parser.add_argument('--normalize_method', default='bayesian',
                        help='method for normalizing')
    args = parser.parse_args()

    assert not (args.only_extractive and args.only_abstractive)
    return args


def get_baseline_data(split_type, split_size, baseline, metric, model):
    data = list()
    if baseline == 'random':
        file = './result_multi/{}/{}/cnndm/data/article_generation_random.npy'.format(
            metric, model)
    else:
        file = './result_multi/{}/{}/cnndm/data/article_generation.npy'.format(
            metric, model)
    model_data = np.load(file, allow_pickle=True)
    split_list = list()
    for j, length_similar_data in enumerate(model_data):
        temp_item_list = list()
        for item in length_similar_data:
            temp_item_list.append(list(item))
        temp_item_list = temp_item_list
        sorted_item_list = sorted(temp_item_list, key=lambda x: x[0])
        split_list.append(sorted_item_list[-1][0])

        # temp_item_list = sorted(temp_item_list, key=lambda x: x[2])
        # index_count = int (len(sorted_item_list) / 10)
        # for index, item in enumerate(sorted_item_list):
        #     item[2] = min(int(index / index_count),9)

        # for index, item in enumerate(sorted_item_list):
        #     item[2] = int(item[2] / 5)

        # sorted_item_list = sorted(temp_item_list, key=lambda x: x[1])
        # index_count = int (len(sorted_item_list) / 10)
        # for index, item in enumerate(sorted_item_list):
        #     item[1] = min(int(index / index_count),9)

        for item in temp_item_list:
            if split_type == 'percent':
                item[0] = j
            elif split_type == 'width':
                item[0] = int(item[0] / split_size)
            item[1] = int(item[1] / 10)
            item[2] = int(item[2] / 10)
            if item[3] > 100:
                item[3] /= 100
            if metric in ['bert_score_f1', 'blanc']:
                item[3] = item[3] + 100
            item[3] = int(item[3]*10)
            # item[3] = int(item[3])

        data.extend(temp_item_list)

    return data, split_list


def get_model_data(data_file: str, args: Namespace) -> Tuple[Set[str], Set[str], Set[str], Dict]:
    summ_ids: Set[str] = set()
    metrics: Set[str] = set()
    articles: Set[str] = set()

    data = dict()
    with open(data_file) as inputf:
        for line in inputf:
            item = json.loads(line)
            curid = item['id']
            summ_id = item['model_id']
            if args.only_extractive and summ_id not in args.extractive_ids:
                continue
            if args.only_abstractive and summ_id not in args.abstractive_ids:
                continue

            summ_ids.add(summ_id)
            articles.add(curid)

            annotations = item[args.annotators_str]
            if curid not in data:
                data[curid] = dict()
                data[curid]['reference'] = item['references']
                text_file: str = item['filepath']
                with open(args.article_dir+'/'+text_file, mode='r', encoding='utf8') as fp:
                    story_content = fp.read()
                    content_raw: str = story_content.split("@highlight")[0]
                    content: str = " ".join(filter(None, [x.strip()
                                                          for x in content_raw.split("\n")]))
                    data[curid]["article"] = content
                data[curid]['article_lens'] = len(
                    data[curid]["article"].split(' '))
                data[curid]['reference_lens'] = len(
                    data[curid]["reference"][0].split(' '))

            annotations = item[args.annotators_str]
            coh = [x["coherence"] for x in annotations]
            con = [x["consistency"] for x in annotations]
            flu = [x["fluency"] for x in annotations]
            rel = [x["relevance"] for x in annotations]

            annotations = defaultdict(float)
            annotations["coherence"] = np.mean(coh)
            annotations["consistency"] = np.mean(con)
            annotations["fluency"] = np.mean(flu)
            annotations["relevance"] = np.mean(rel)
            annotations['all'] = np.mean(
                coh)+np.mean(con)+np.mean(flu)+np.mean(rel)

            data[curid][summ_id] = dict()
            data[curid][summ_id]['generation'] = item['decoded']
            data[curid][summ_id]['generation_lens'] = len(
                data[curid][summ_id]['generation'].split())
            data[curid][summ_id]['annotation_score'] = annotations
            data[curid][summ_id]['automatic_score'] = defaultdict(float)

            scores = item[f'metric_scores_{args.subset}']
            for key1, val1 in scores.items():
                if key1 == "id":
                    continue
                # supert returned a list of length 1
                if key1 == "supert":
                    data[curid][summ_id]['automatic_score'][key1] = val1[0]
                    metrics.add(key1)
                elif key1 == "rouge":
                    for key2, val2 in scores["rouge"].items():
                        data[curid][summ_id]['automatic_score'][key2] = val2
                        metrics.add(key2)
                else:
                    data[curid][summ_id]['automatic_score'][key1] = val1
                    metrics.add(key1)

            scores = item['metric_scores_1']
            for key1, val1 in scores.items():
                if key1 == "id":
                    continue
                # supert returned a list of length 1
                if key1 == "supert" and key1 not in data[curid][summ_id]['automatic_score']:
                    data[curid][summ_id]['automatic_score'][key1] = val1[0]
                    metrics.add(key1)
                elif key1 == "rouge":
                    for key2, val2 in scores["rouge"].items():
                        if key2 not in data[curid][summ_id]['automatic_score']:
                            data[curid][summ_id]['automatic_score'][key2] = val2
                            metrics.add(key2)
                else:
                    if key1 not in data[curid][summ_id]['automatic_score']:
                        data[curid][summ_id]['automatic_score'][key1] = val1
                        metrics.add(key1)

    return summ_ids, metrics, articles, data


def get_correlation_score(correlator, data: Dict, metrics: Set[str], summ_ids: Set[str]) -> Dict[str, List[float]]:
    result = dict()
    for metric in metrics:
        coherence_scores, consistency_scores, fluency_scores, relevance_scores = [], [], [], []
        metric_scores = []
        for summ_id in summ_ids:
            cur_metric = []
            cur_coherence, cur_consistency, cur_fluency, cur_relevance = [], [], [], []
            for _, item in data.items():
                cur_metric.append(item[summ_id]['automatic_score'][metric])
                cur_coherence.append(
                    item[summ_id]['annotation_score']["coherence"])
                cur_consistency.append(
                    item[summ_id]['annotation_score']["consistency"])
                cur_fluency.append(
                    item[summ_id]['annotation_score']["fluency"])
                cur_relevance.append(
                    item[summ_id]['annotation_score']["relevance"])

            metric_scores.append(np.mean(cur_metric))
            coherence_scores.append(np.mean(cur_coherence))
            consistency_scores.append(np.mean(cur_consistency))
            fluency_scores.append(np.mean(cur_fluency))
            relevance_scores.append(np.mean(cur_relevance))

        coherence_corr = correlator(coherence_scores, metric_scores)[0]
        consistency_corr = correlator(consistency_scores, metric_scores)[0]
        fluency_corr = correlator(fluency_scores, metric_scores)[0]
        relevance_corr = correlator(relevance_scores, metric_scores)[0]

        coherence_corrs_final = format(round(coherence_corr, 4), ".4f")
        consistency_corrs_final = format(round(consistency_corr, 4), ".4f")
        fluency_corrs_final = format(round(fluency_corr, 4), ".4f")
        relevance_corrs_final = format(round(relevance_corr, 4), ".4f")

        corr_list = [coherence_corrs_final, consistency_corrs_final,
                     fluency_corrs_final, relevance_corrs_final]

        result[metric] = corr_list
    return result


def normalize_data(args, map_function_scale_list: List[str], data, summ_id_index, summ_id, metric, model_dict: Dict[str, Union[BayesianNetwork, LinearRegression, List[int]]]):
    result = dict()
    if args.normalize_method == 'bayesian':
        model = BayesianNetwork().from_dict(model_dict['model'])
    elif args.normalize_method == 'linear':
        model = model_dict['model']
    split_list = model_dict['split_list']

    summary_item_list = list()
    reference_item_list = list()
    id_list = list()
    for id, item in data.items():
        id_list.append(id)
        article_length = item['article_lens']
        article_length_index = 0

        if args.split_type == 'percent':
            while article_length > split_list[article_length_index]:
                article_length_index += 1
        else:
            article_length_index = int(article_length / args.split_size)

        reference_length = int(item['reference_lens'] / 10)
        generation_length = int(item[summ_id]['generation_lens'] / 10)

        if args.normalize_method == 'bayesian':
            summary_item_list.append(
                [summ_id_index, article_length_index, reference_length, generation_length, None])
            reference_item_list.append(
                [summ_id_index, article_length_index, reference_length, reference_length, None])
        elif args.normalize_method == 'linear':
            summary_item_list.append(
                [summ_id_index, article_length_index, reference_length, generation_length])
            reference_item_list.append(
                [summ_id_index, article_length_index, reference_length, reference_length])

    if args.debug:
        summary_item_list = summary_item_list[:5]
        reference_item_list = reference_item_list[:5]
        id_list = id_list[:5]

    summary_predictd_result = list()
    reference_predictd_result = list()

    if args.normalize_method == 'bayesian':
        summary_predicted_prob = model.predict_proba(
            summary_item_list, check_input=False)
        for it in summary_predicted_prob:
            score = 0
            for k, v in it[-1].parameters[0].items():
                score += k*v
            summary_predictd_result.append(score)

        reference_predicted_prob = model.predict_proba(
            reference_item_list, check_input=False)
        for it in reference_predicted_prob:
            score = 0
            for k, v in it[-1].parameters[0].items():
                score += k*v
            reference_predictd_result.append(score)
    elif args.normalize_method == 'linear':
        summary_predictd_result = model.predict(summary_item_list)
        reference_predictd_result = model.predict(reference_item_list)

    for map_function_scale in map_function_scale_list:
        result[map_function_scale] = dict()
    for j, id in enumerate(id_list):
        score = data[id][summ_id]['automatic_score'][metric]
        for map_function_scale in map_function_scale_list:
            if id not in result[map_function_scale]:
                result[map_function_scale][id] = dict()
            adjusted_score = map_score(map_function_scale,
                                       score, summary_predictd_result[j], reference_predictd_result[j])
            result[map_function_scale][id][metric] = adjusted_score
    return summ_id, metric, result


def map_score(map_function_scale: Union[int, str], score: float, summary_score: float, reference_score: float):
    if summary_score == 0:
        summary_score = 1e-2
    if map_function_scale == 'no':
        adjusted_score = score
    else:
        alpha = pow(2, map_function_scale)
        adjusted_score = alpha * \
            (reference_score - summary_score) / summary_score * score + score
    return adjusted_score


def create_bayesian_model(args, analyze_info: AnalyzeCorrelationInfo, metric, folder: str) -> Tuple[BayesianNetwork, List[int]]:
    random_data = list()
    for i, model in enumerate(analyze_info.summ_ids):
        model_data, split_list = get_baseline_data(
            args.split_type, args.split_size, args.baseline, metric, model)
        # for item in model_data[:300]:
        for item in model_data:
            random_data.append([i]+item)
    key_list = ['model', 'article length',
                'groundtruth length', 'generation length', 'score']
    # edge_list = [(3, 4), (0, 4), (0, 3), (1,3)]
    edge_list = [(3, 4), (0, 4), (0, 3)]
    exclude_edges = [(4, 0), (4, 1), (4, 2), (4, 3), (3, 0),
                     (3, 1), (3, 2), (2, 0), (2, 1), (1, 0), (0, 1), (0, 2)]
    ba_model = BayesianNetwork.from_samples(
        np.array(random_data), state_names=key_list, include_edges=edge_list, exclude_edges=exclude_edges)
    plt_folder = os.path.join(os.path.join(folder, 'picture'), 'model')
    if not os.path.exists(plt_folder):
        os.makedirs(plt_folder)
    plt_file = os.path.join(plt_folder, "{}.pdf".format(metric))
    ba_model.plot(plt_file)
    ba_model = ba_model.to_dict()
    return ba_model, split_list


def create_linear_model(args, analyze_info: AnalyzeCorrelationInfo, metric, folder: str) -> Tuple[LinearRegression, List[int]]:
    random_data = list()
    for i, model in enumerate(analyze_info.summ_ids):
        model_data, split_list = get_baseline_data(
            args.split_type, args.split_size, args.baseline, metric, model)
        for item in model_data:
            random_data.append([i]+item)

    model = make_pipeline(StandardScaler(with_mean=False), LinearRegression())
    data = np.array(random_data)
    x_data = data[:, :-1]
    y_data = data[:, -1]
    model.fit(x_data, y_data)
    return model, split_list


def create_trans_model(args, analyze_info: AnalyzeCorrelationInfo, folder: str) -> Dict[str, Union[BayesianNetwork, List[int], LinearRegression]]:
    network_dict = dict()
    for metric in tqdm.tqdm(analyze_info.sorted_keys, desc='create network', ncols=100):
        if args.normalize_method == 'bayesian':
            model, split_list = create_bayesian_model(
                args, analyze_info, metric, folder)
        elif args.normalize_method == 'linear':
            model, split_list = create_linear_model(
                args, analyze_info, metric, folder)
        network_dict[metric] = {'model': model, 'split_list': split_list}

    return network_dict


def calculate_new_score(args, analyze_info, map_function_scale_list, data, model_dict):
    pbar = tqdm.tqdm(total=len(analyze_info.sorted_keys)
                     * len(analyze_info.summ_ids), ncols=100, desc='normalize score')
    update = lambda *args: pbar.update()
    new_data = dict()
    res_list = list()

    if args.debug and args.single_process:
        for metric in analyze_info.sorted_keys:
            for summ_id_index, summ_id in enumerate(analyze_info.summ_ids):
                res_list.append(normalize_data(args, map_function_scale_list, data,
                                summ_id_index, summ_id, metric, model_dict[metric]))
                pbar.update()
    else:
        with Pool(processes=15) as pool:
            for metric in analyze_info.sorted_keys:
                for summ_id_index, summ_id in enumerate(analyze_info.summ_ids):
                    res_list.append(pool.apply_async(normalize_data, (args, map_function_scale_list, data,
                                    summ_id_index, summ_id, metric, model_dict[metric],), callback=update))
            pool.close()
            pool.join()

    for res in res_list:
        if args.debug and args.single_process:
            summ_id, metric, result = res
        else:
            summ_id, metric, result = res.get()
        for map_function_method, map_function_data in result.items():
            if map_function_method not in new_data:
                new_data[map_function_method] = copy.deepcopy(data)
            for id, id_data in map_function_data.items():
                if 'automatic_score_old' not in new_data[map_function_method][id][summ_id]:
                    new_data[map_function_method][id][summ_id]['automatic_score_old'] = dict(
                    )
                new_data[map_function_method][id][summ_id]['automatic_score_old'][
                    metric] = new_data[map_function_method][id][summ_id]['automatic_score'][metric]
                new_data[map_function_method][id][summ_id]['automatic_score'][metric] = id_data[metric]
    pbar.close()
    sys.stdout.flush()

    return new_data


def compare_correlation(analyze_info: AnalyzeCorrelationInfo, map_function_scale_list: str):
    args = get_params()
    baseline = args.baseline
    _, metrics, articles, data = get_model_data(args.input_file, args)
    summ_ids = analyze_info.summ_ids
    split_size = int(args.split_size)

    print('{} {} {}'.format(baseline, args.split_type, split_size))

    folder = './normalize/{}/{}/{}/{}'.format(
        args.normalize_method, args.split_type, baseline, split_size)
    if not os.path.exists(folder):
        os.makedirs(folder)
    correlator_dict = {'pearsonr': pearsonr, 'kendallatu': kendalltau}
    old_result = dict()
    for correlator_name, correlator in correlator_dict.items():
        correlator_score_folder = os.path.join(
            os.path.join(folder, 'score'), correlator_name)
        if not os.path.exists(correlator_score_folder):
            os.makedirs(correlator_score_folder)
        old_result[correlator_name] = get_correlation_score(
            correlator, data, analyze_info.sorted_keys, analyze_info.summ_ids)
        with open(os.path.join(correlator_score_folder, 'original_result.csv'), mode='w', encoding='utf8') as fp:
            fp.write("Metrics,Coherence,Consistency,Fluency,Relevance\n")
            for k, v in old_result[correlator_name].items():
                fp.write('{},{}\n'.format(k, ','.join(v)))

    network_dict = create_trans_model(args, analyze_info, folder)

    new_data = calculate_new_score(
        args, analyze_info, map_function_scale_list, data, network_dict)

    data_folder = os.path.join(folder, 'data')
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    with open(os.path.join(data_folder, 'score.json'), mode='w', encoding='utf8') as fp:
        json.dump(new_data, fp)

    record_dict = dict()
    pbar = tqdm.tqdm(total=len(map_function_scale_list)
                     * len(correlator_dict), ncols=100, desc='record result')
    for correlator_name, correlator in correlator_dict.items():
        record_dict[correlator_name] = list()
        correlator_score_folder = os.path.join(
            os.path.join(folder, 'score'), correlator_name)
        for map_function_scale, data in new_data.items():
            result = get_correlation_score(
                correlator, data, analyze_info.sorted_keys, analyze_info.summ_ids)
            with open(os.path.join(correlator_score_folder, 'result_{}.csv'.format(map_function_scale)), mode='w', encoding='utf8') as fp:
                fp.write("Metrics,Coherence,Consistency,Fluency,Relevance\n")
                for k, v in result.items():
                    score_list = [float(item) for item in v]
                    old_score_list = [float(item)
                                      for item in old_result[correlator_name][k]]
                    full_score_list = list()
                    for score, old_score in zip(score_list, old_score_list):
                        full_score_list.append(
                            "{:.4f} ({:.4f})".format(score, score - old_score))
                    fp.write('{},{}\n'.format(k, ','.join(full_score_list)))

            count_dict = {'positive':0, 'tier':0, 'negative':0}
            mean_difference = list()
            coherence_count_dict = {'positive':0, 'tier':0, 'negative':0}
            coherence_mean_difference = list()

            for key, value in result.items():
                for index, _ in enumerate(value):
                    if float(value[index]) > float(old_result[correlator_name][key][index]):
                        count_dict['positive']+=1
                    elif float(value[index]) < float(old_result[correlator_name][key][index]):
                        count_dict['negative']+=1
                    else:
                        count_dict['tier']+=1
                    mean_difference.append(float(value[index]) - float(old_result[correlator_name][key][index]))
                if float(value[0]) > float(old_result[correlator_name][key][0]):
                    coherence_count_dict['positive']+=1
                elif float(value[0]) < float(old_result[correlator_name][key][0]):
                    coherence_count_dict['negative']+=1
                else:
                    coherence_count_dict['tier']+=1
                coherence_mean_difference.append(float(value[0]) - float(
                    old_result[correlator_name][key][0]))
            record = "{},{},{},{},{},{:.4f},{},{},{},{:.4f}".format(
                correlator_name, map_function_scale, count_dict['positive'],count_dict['negative'],count_dict['tier'], np.mean(mean_difference), coherence_count_dict['positive'],coherence_count_dict['negative'],coherence_count_dict['tier'], np.mean(coherence_mean_difference))
            record_dict[correlator_name].append(record)
            pbar.update()

        with open(os.path.join(correlator_score_folder, 'result_conclusion.csv'), mode='w', encoding='utf8') as fp:
            fp.write(
                'correlator_name, map_function_scale, count positive, count negative, count tier, mean improvement, coherence count positive, coherence count negative, coherence count tier, coherent mean improvement\n')
            for record in record_dict[correlator_name]:
                fp.write(record+'\n')
    pbar.close()

    for _, record_list in record_dict.items():
        for record in record_list:
            print(record)


def main():
    # baseline = sys.argv[1]
    # baseline = 'random'
    map_function_scale_list = ['no', -4, -3, -2, -1, 0, 1, 2, 3, 4]
    random.seed(42)
    analyze_info = AnalyzeCorrelationInfo()
    compare_correlation(analyze_info, map_function_scale_list)


if __name__ == '__main__':
    main()
