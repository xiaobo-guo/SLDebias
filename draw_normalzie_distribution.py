import json
import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, kendalltau
import copy
from collections import defaultdict
import tqdm

metrics_list = ['bert_score_f1', 'blanc', 'bleu', 'chrf', 'meteor', 'mover_score', 'rouge_1_f_score', 'rouge_2_f_score',
                'rouge_3_f_score', 'rouge_4_f_score', 'rouge_l_f_score', 'rouge_su*_f_score', 'rouge_w_1.2_f_score',  'rouge_we_3_f']
names_list = ['BertScore', 'BLANC', 'BLEU', 'chrF', 'METEOR', 'MoverScore', 'ROUGE-1',
              'ROUGE-2', 'ROUGE-3', 'ROUGE-4', 'ROUGE-L', 'ROUGE-su*', 'ROUGE-w', 'ROUGE-we-3']
models_list = ['M1', 'M5', 'M8', 'M9', 'M10', 'M11',
                         'M12', 'M13', 'M14', 'M15', 'M17', 'M20', 'M22', 'M23']

def read_data():
    file = './normalize/bayesian/percent/prediction/10/data/score.json'
    with open(file,mode='r',encoding='utf8') as fp:
        data = json.load(fp)

    annotation_score_list = dict()
    automatic_score_list = dict()
    id_list = list()

    for item_id, item_data in data['no'].items():
        id_list.append(item_id)
        for model_name, model_data in item_data.items():
            if model_name not in models_list:
                continue
            if model_name not in automatic_score_list:
                annotation_score_list[model_name] = list()
                automatic_score_list[model_name] = dict()
                for metric in metrics_list:
                    automatic_score_list[model_name][metric] = list()
            annotation_score_list[model_name].append(model_data['annotation_score']['coherence'])
            for metric in metrics_list:
                automatic_score_list[model_name][metric].append(model_data['automatic_score_old'][metric])
    result_dict = dict()
    for model_name, model_data in automatic_score_list.items():
        result_dict[model_name] = dict()
        for metric, metric_data in model_data.items():
            result_dict[model_name][metric] = dict()
            pearsonr_score = pearsonr(annotation_score_list[model_name], metric_data)[0]
            kendalltau_score = kendalltau(annotation_score_list[model_name], metric_data)[0]
            result_dict[model_name][metric] = {'pearsonr':pearsonr_score,'kendalltau':kendalltau_score}

    pbar = tqdm.tqdm(total=len(data)*100*len(models_list)*len(metrics_list),ncols=100)

    # record = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:defaultdict(list))))
    record = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for scale_value, scale_data in data.items():
        if scale_value == "no":
            continue
        for item_id, item_data in scale_data.items():
            for model_name, model_data in item_data.items():
                if model_name not in models_list:
                    continue
                human_annotation_data = annotation_score_list[model_name]
                for metric in metrics_list:
                    original_correlation_score = result_dict[model_name][metric]
                    metric_score = copy.deepcopy(automatic_score_list[model_name][metric])
                    change_index = id_list.index(item_id)
                    metric_score[change_index] = model_data['automatic_score'][metric]
                    pearsonr_score = pearsonr(human_annotation_data, metric_score)[0]
                    kendalltau_score = kendalltau(human_annotation_data, metric_score)[0]
                    record_item = dict()

                    record_item['pearsonr'] = pearsonr_score - original_correlation_score['pearsonr']
                    record_item['kendalltau'] = pearsonr_score - original_correlation_score['kendalltau']
                    record_item['difference'] = model_data['automatic_score'][metric] - model_data['automatic_score_old'][metric]
                    record[scale_value][model_name][metric].append(record_item)
                    pbar.update()
    pbar.close()
    return record


def draw_picture(scale_value, model_name, metric, data):
    sorted_data = sorted(data, key=lambda x:x['difference'])
    for rank in ['pearsonr','kendalltau']:
        folder = './normalize/bayesian/percent/prediction/10/picture/distritbution/{}'.format(rank)
        if not os.path.exists(folder):
            os.makedirs(folder)
        plot_data = {'x':[],'y':[],'c':[]}
        for index, item in enumerate(sorted_data):
            plot_data['x'].append(item['difference'])
            plot_data['y'].append(index)

            if item[rank] > 0:
                plot_data['c'].append('r')
            elif item[rank] < 0:
                plot_data['c'].append('b')
            else:
                plot_data['c'].append('g')
        plt.scatter(plot_data['x'],plot_data['y'],c=plot_data['c'])
        plt.axvline(x=0)
        plt.savefig(os.path.join(folder,'{}_{}_{}.pdf'.format(scale_value, model_name, metric)))
        plt.close()


def main():
    record = read_data()
    for scale_value, scale_data in tqdm.tqdm(record.items(),ncols=100):
        for model_name, model_data in scale_data.items():
            for metric, metric_data in model_data.items():
                draw_picture(scale_value,model_name,metric,metric_data)
if __name__ == '__main__':
    main()
