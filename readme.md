# Length Does Matter: Summary Length can Bias Summarization Metrics

This repo covers implementations Length Does Matter: Summary Length can Bias Summarization Metrics by Xiaobo Guo and [Soroush Vosoughi](https://www.cs.dartmouth.edu/~soroush/). The paper will be published at EMNLP 2023.

## Setup
The code is prepared for Python 3.8.10.

To run the code, please run the following code for installing packages:

```
pip install -r ./requirements.txt
git clone https://github.com/Yale-LILY/SummEval.git
cd evaluation
pip install -e .
```

Please run the following command and add it to your startup script:

```
export ROUGE_HOME=/$Home/SLDebias/SummEval/evaluation/summ_eval/ROUGE-1.5.5/
export PYTHONPATH=$PYTHONPATH:/$Home/SLDebias/SummEval/evaluation/summ_eval
```

Please also run these commands:

```
pip install -U  git+https://github.com/bheinzerling/pyrouge.git
sudo apt install openjdk-8-jdk
```


## Prepare Data

You must prepare the data for three datasets: CNN/DailyMail, WikiHow, and Webis-TLDR-17. We will introduce the preparation of them one by one.

### CNN/DailyMail 
For the CNN/DailyMail dataset, you can download it from [here](https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail), unzip the file, and put it in the folder: 

> ./data

The file should be in the format of 
```
./data/cnn_dailymail/train.csv
./data/cnn_dailymail/validation.csv
./data/cnn_dailymail/test.csv
```

### WikiHow
For the WikiHow dataset, you can download it from [here](https://ucsb.app.box.com/s/ap23l8gafpezf4tq3wapr6u8241zz358), and put the files wikihowAll.csv, all_test.txt, all_train.txt, and all_val.txt in the folder:

> ./data/original

Then run the following command:

```
python ./util/prepare_data.py wikihowAll.csv
```

### Webis-TLDR-17 
For the Webis-TLDR-17 dataset, you can download from [here](https://webis.de/data/webis-tldr-17.html), unzip it and put json file corpus-webis-tldr-17.json in the folder:

> ./data/orginal

Then run the following command:

```
python ./util/prepare_data.py corpus-webis-tldr-17.json
```


### Adjustment Experiments
For the experiments of score adjustment, please follow the instructions of the [SummEval](https://github.com/Yale-LILY/SummEval/blob/master/README.md) for preparing the data.


## Fine-tune Model

In our paper, we fine-tune three models on 3 datasets. We suggest using the predictions provided by us for metrics analysis but we also provide the code for fine-tuning the model yourself.

For fine-tuning the model, please run the following code:
```
./shell/fine_tune_model.sh
```

Noting that are 9 models trained in the shell file, you can also split them to fine-tune the models parallelly.

## Analyze Metrics Trends
We provide the prediction results in the following path:
```
./model/{dataset_name}/{model_name}/0/generated_predictions.txt
```
The results are predicted with our fine-tuned model.

For conducting the analysis, please run the following command:
```
./shell/trend_analysis.sh
```

Then run the following command
```
python ./util/draw_picture.py
```
to generate the figure of trends reported in our paper in the folder 'picture'

Run the following command
```
python ./bayesian_network_analysis.py size auto
```
to generate the Table of the Bayesian network analysis reported in our paper in the folder ''ba_model''. 

## Score Adjustment

For the experiments of the Score Adjustment, please run the following command:
```
./shell/score_adjustment.sh
```

Then to generate the comparison results of the adjusted and original score for the random method, Bayesian network, and linear regression, please run the following commands:
```
python ./run_random_baseline.py
python ./normalize_score_random.py --baseline=prediction --split_type=percent --split_size=10 --normalize_method=random
python ./normalize_score.py --baseline=prediction --split_type=percent --split_size=10 --normalize_method=bayesian
python ./normalize_score.py --baseline=prediction --split_type=percent --split_size=10 --normalize_method=linear
```

To generate the Figure of the difference between the Bayesian network and the Linear regression please run the command:
```
python ./util/draw_normalize.py
```
