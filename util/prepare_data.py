import os
import sys
import json
import random
from typing import Dict, List, Union
import tqdm
import pandas as pd
import random
import numpy as np
import re
import csv
import re


class DataPreparer(object):
    def __init__(self, dataset: str, content_name, summary_name) -> None:
        self.original_data_file = './data/original/{}'.format(dataset)
        self.dataset = dataset.split('.')[0]
        self.target_data_fodler = './data/{}'.format(self.dataset)
        self.content_name = content_name
        self.summary_name = summary_name

    def read_data(self) -> List[Dict[str, Union[str, int]]]:
        if 'json' in self.original_data_file:
            return self._read_data_json()
        elif 'csv' in self.original_data_file:
            return self._read_data_csv()

    def _read_data_json(self) -> List[Dict[str, Union[str, int]]]:
        data = list()
        with open(self.original_data_file, mode='r', encoding='utf8') as fp:
            for line in tqdm.tqdm(fp.readlines(), ncols=100, desc='Read data'):
                item = json.loads(line.strip())
                content = item[self.content_name].replace(
                    '\n\n', '\n').replace('\n', ' ').replace('\r', 'r')
                summary = item[self.summary_name].replace(
                    '\n\n', '\n').replace('\n', ' ').replace('\r', 'r')
                data.append({'article': content, 'highlights': summary})
        return data

    def _read_data_csv(self) -> List[Dict[str, Union[str, int]]]:
        data = list()
        original_data = pd.read_csv(self.original_data_file)
        json_data = json.loads(original_data.to_json(orient='records'))
        for item in tqdm.tqdm(json_data, ncols=100, desc='Read data'):
            if item[self.content_name] is not None and item[self.summary_name] is not None:
                content = item[self.content_name].replace(
                    '\n\n', '\n').replace('\n', ' ').replace('\r', 'r')
                summary = item[self.summary_name].replace(
                    '\n\n', '\n').replace('\n', ' ').replace('\r', 'r')
                data.append({'article': content, 'highlights': summary})
                if 'title' in item:
                    data[-1]['title'] = item['title']
        return data

    def process_data(self, data: List[Dict[str, Union[str, int]]]) -> Dict[str, pd.DataFrame]:
        processed_data = list()
        if self.dataset == 'corpus-webis-tldr-17':
            processed_data = self.process_data_webis(data)
            total_lens = len(processed_data)
            train_number = int(0.7*total_lens)
            valid_numbr = int(0.9*total_lens)
            random.shuffle(processed_data)

            train_data = processed_data[:train_number]
            validation_data = processed_data[train_number:valid_numbr]
            test_data = processed_data[valid_numbr:]

        elif self.dataset == 'wikihowSep' or self.dataset == 'wikihowAll':
            train_data, validation_data, test_data = self.process_data_wikiHow(
                data)

        train_data = pd.DataFrame([[item['article'], item['highlights']]
                                  for item in train_data], columns=['article', 'highlights'])
        validation_data = pd.DataFrame([[item['article'], item['highlights']]
                                       for item in validation_data], columns=['article', 'highlights'])
        test_data = pd.DataFrame([[item['article'], item['highlights']]
                                 for item in test_data], columns=['article', 'highlights'])

        data = {'train': train_data,
                'validation': validation_data, 'test': test_data}
        return data

    def process_data_webis(self, data: List[Dict[str, Union[str, int]]]) -> Dict[str, pd.DataFrame]:
        error_count = 0
        filtered_data = list()
        summary_length = list()
        content_length = list()
        article_set = set()
        summary_set = set()
        new_data = list()
        for item in tqdm.tqdm(data, ncols=100, desc='Filter Duplicated Data'):

            temp = item['article'].lower()
            split_word = 't[i,l]?ld?[^a-z]{0,3}dr'
            index_list = re.findall(split_word, temp)
            index = -1
            if 'til dr' in index_list:
                index_list.remove('til dr')
            if 'tildr' in index_list:
                index_list.remove('tildr')
            if 'tl;dr' in index_list:
                index_list = ['tl;dr']
            elif 'tldr' in index_list:
                index_list = ['tldr']
            elif 'tl dr' in index_list:
                index_list = ['tl dr']
            if len(index_list) == 1:
                index = temp.find(index_list[0])
            elif 'ntl;sdr' in temp:
                index = temp.find('ntl;sdr')
            elif 'stl;sdr' in temp:
                index = temp.find('stl;sdr')
            if index == -1:
                error_count += 1
                continue
            item['article'] = item['article'][:index]
            if item['article'] in article_set or item['highlights'] in summary_set or item['highlights'] == 'null':
                continue
            new_data.append(item)
            article_set.add(item['article'])
            summary_set.add(item['highlights'])
        print(error_count)
        data = new_data
        for item in tqdm.tqdm(data, ncols=100, desc='Get length'):
            summary_length.append(len(item['highlights'].split(' ')))
            content_length.append(len(item['article'].split(' ')))

        summary_mean = np.mean(summary_length)
        summary_std = np.std(summary_length)
        content_mean = np.mean(content_length)
        content_std = np.std(content_length)
        for item in tqdm.tqdm(data, ncols=100, desc='Filter data'):
            if (len(item['highlights'].split(' ')) > (summary_mean - summary_std) and len(item['highlights'].split(' ')) < (summary_mean + summary_std)) and (len(item['article'].split(' ')) > (content_mean - content_std) and len(item['article'].split(' ')) < (content_mean + content_std)):
                filtered_data.append(item)
        random.shuffle(filtered_data)
        return filtered_data

    def process_data_wikiHow(self, data: List[Dict[str, Union[str, int]]]) -> Dict[str, pd.DataFrame]:
        article_set = set()
        summary_set = set()
        new_data = list()
        for item in tqdm.tqdm(data, ncols=100, desc='Filter Duplicated Data'):
            if item['article'] in article_set or item['highlights'] in summary_set:
                continue
            if len(item['highlights']) >= (0.75*len(item['article'])):
                continue

            item['highlights'] = item['highlights'].replace(".,", ".").strip()
            item['article'] = re.sub(
                r'[.]+[\n]+[,]', ".\n", item['article']).strip()
            new_data.append(item)
            article_set.add(item['article'])
            summary_set.add(item['highlights'])

        data = new_data

        splited_title_dict = dict()
        for file_type in ['train', 'val', 'test']:
            splited_title_list = list()
            title_list_file = 'data/original/all_{}.txt'.format(file_type)
            with open(title_list_file, mode='r', encoding='utf8') as fp:
                for line in fp.readlines():
                    splited_title_list.append(line.strip())
            splited_title_dict[file_type] = set(splited_title_list)

        train_data = list()
        eval_data = list()
        test_data = list()

        for item in new_data:
            if item['title'].replace(' ', '') in splited_title_dict['train']:
                train_data.append(item)
            elif item['title'].replace(' ', '') in splited_title_dict['val']:
                eval_data.append(item)
            elif item['title'].replace(' ', '') in splited_title_dict['test']:
                test_data.append(item)
        return train_data, eval_data, test_data

    def write_data(self, processed_data: Dict[str, pd.DataFrame]) -> None:
        if not os.path.exists(self.target_data_fodler):
            os.makedirs(self.target_data_fodler)
        for k, data in processed_data.items():
            file = '{}/{}.csv'.format(self.target_data_fodler, k)
            data.to_csv(file, index=False)


def prepare_data(dataset: str):
    original_file = './data/original/{}'.format(dataset)


def main():
    seed = 42
    random.seed(seed)

    dataset = sys.argv[1]
    if dataset == 'corpus-webis-tldr-17.json':
        content_name = 'body'
        summary_name = 'summary'
    elif dataset == 'wikihowAll.csv':
        content_name = 'text'
        summary_name = 'headline'

    data_preparer = DataPreparer(dataset, content_name, summary_name)
    data = data_preparer.read_data()
    processed_data = data_preparer.process_data(data)
    data_preparer.write_data(processed_data)


if __name__ == '__main__':
    main()
