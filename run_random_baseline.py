import json
import os
import numpy as np
from nltk import sent_tokenize
import random
import transformers
import logging
from transformers import set_seed
import tqdm
from bert_score import BERTScorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


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


length_dict = dict()
model_list = ['M1', 'M5', 'M8', 'M9', 'M10', 'M11',
                         'M12', 'M13', 'M14', 'M15', 'M17', 'M20', 'M22', 'M23']

length_list = [44,50,56,62,68,74,80,86]

file = './data/cnndm/human/model_annotations.aligned.jsonl'



def get_sumamry():
    data = dict()
    for summary_length in length_list:
        data[summary_length] = dict()

    with open(file,mode='r',encoding='utf8') as fp:
        for line in fp.readlines():
            item = json.loads(line.strip())
            id = item['id']
            if id in data[length_list[0]]:
                continue
            reference = item['references'][0]
            summary = item['decoded']
            text_path = os.path.join('./data',item['filepath'])
            temp_text_list = list()
            with open(text_path,mode='r',encoding='utf8') as textfp:
                for line in textfp.readlines():
                    if line.strip()!='':
                        temp_text_list.append(line.strip())
            sent_list = list()
            for text in temp_text_list:
                sent_list.extend(sent_tokenize(text))
            article = ' '.join(sent_list)
            random.shuffle(sent_list)
            for summary_length in length_list:
                summary = random_extract(sent_list, summary_length)
                data[summary_length][id] = {'summary':summary,'reference':reference,'article':article}
    return data


def calculate_score(data):
    record = dict()
    for length_limit, length_data in tqdm.tqdm(data.items(),dynamic_ncols=True,desc='Calculate Score',leave=False):
        score_dict = calculate_score_item(length_data)
        record[length_limit] = score_dict
    return record
        

def calculate_score_item(data):
    article_list = list()
    reference_list = list()
    summary_list = list()
    article_lens_list = list()
    reference_lens_list = list()
    summary_lens_list = list()
    
    for _, v in data.items():
        for item in v:
            article_list.append(item['article'])
            summary_list.append(item['summary'])
            reference_list.append(item['reference'])
            article_lens_list.append(len(item['article'].split()))
            summary_lens_list.append(len(item['summary'].split()))
            reference_lens_list.append(len(item['reference'].split()))


    score_list = dict()

    datastats_model = DataStatsMetric()

    score_dict = datastats_model.evaluate_batch(
        summaries=summary_list, input_texts=article_list, aggregate=False, show_progress_bar=True)

    chencherry = SmoothingFunction()
    rouge_model = RougeMetric()
    rougewe_model = RougeWeMetric()
    meteor_model = MeteorMetric()
    chrfpp_model = ChrfppMetric()
    blanc_model = BlancMetric(inference_batch_size=64, finetune_batch_size=12)
    supert_model = SupertMetric()

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

    score_dict = rougewe_model.evaluate_batch(
        summaries=summary_list, references=reference_list, aggregate=False, show_progress_bar=True)
    for i, score in enumerate(score_dict):
        for k, v in score.items():
            if k.endswith('_f'):
                if k not in score_list:
                    score_list[k] = list()
                score_list[k].append(
                    (article_lens_list[i], reference_lens_list[i], summary_lens_list[i], v*100))

    score_dict = meteor_model.evaluate_batch(
        summaries=summary_list, references=reference_list, aggregate=False, show_progress_bar=True)
    for i, score in enumerate(score_dict):
        for k, v in score.items():
            if k not in score_list:
                score_list[k] = list()
            score_list[k].append(
                (article_lens_list[i], reference_lens_list[i], summary_lens_list[i], v*100))

    score_dict = chrfpp_model.evaluate_batch(
        summaries=summary_list, references=reference_list, aggregate=False, show_progress_bar=True)
    for i, score in enumerate(score_dict):
        for k, v in score.items():
            if k not in score_list:
                score_list[k] = list()
            score_list[k].append(
                (article_lens_list[i], reference_lens_list[i], summary_lens_list[i], v))

    data_temp = list()
    for _, v in data.items():
        for item in v:
            data_temp.append(item)
    for item in tqdm.tqdm(data_temp,desc='calculate BLEU',dynamic_ncols=True,leave=False):
        groundtruth = item['reference']
        summary = item['summary']
        bleu = sentence_bleu(references=[groundtruth.replace('\n', ' ').split(
        )], hypothesis=summary.replace('\n', ' ').split(), smoothing_function=chencherry.method1)
        if 'bleu' not in score_list:
            score_list['bleu'] = list()
        score_list['bleu'].append((0,0,0,bleu*100))


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


    score_dict = blanc_model.evaluate_batch(
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
                
    moverscore_model = MoverScoreMetric()
    Bertscore_model = BERTScorer(lang="en", rescale_with_baseline=True)

    score_dict = moverscore_model.evaluate_batch(
        summaries=summary_list, references=reference_list, aggregate=False, show_progress_bar=True)
    for i, score in enumerate(score_dict):
        for k, v in score.items():
            if k not in score_list:
                score_list[k] = list()
            score_list[k].append(
                (article_lens_list[i], reference_lens_list[i], summary_lens_list[i], v*100))

    _, _, bert_score = Bertscore_model.score(
        cands=summary_list, refs=reference_list, batch_size=12)
    if 'bert_score_f1' not in score_list:
        score_list['bert_score_f1'] = list()
    for i, score in enumerate(bert_score.numpy().tolist()):
        score_list['bert_score_f1'].append(
            (article_lens_list[i], reference_lens_list[i], summary_lens_list[i], score*100))




    for k, v in score_list.items():
        s_list = [item[3] for item in v]
        score_list[k] = np.mean(s_list)

    return score_list


def random_extract(sent_list, length_limit):
    summary = list()
    word_count = 0
    for sent in sent_list:
        if word_count + len(sent.split(' '))< length_limit:
            summary.append(sent)
            word_count += len(sent.split(' '))
    return ' '.join(summary)

def main():
    data = dict()
    for random_seed in range(10):
        set_seed(random_seed)
        single_summary = get_sumamry()
        for length_limit, length_data in single_summary.items():
            if length_limit not in data:
                data[length_limit] = dict()
            for k, v in length_data.items():
                if k not in data[length_limit]:
                    data[length_limit][k] = list()
                data[length_limit][k].append(v)

    score = calculate_score(data)
    with open('random_baseline.json',mode='w',encoding='utf8') as fp:
        fp.write(json.dumps(score))

if __name__ == '__main__':
    main()