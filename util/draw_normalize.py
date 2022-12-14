import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

model_list = ['bayesian','linear']
score_type = 'kendallatu'

data = dict()
for model in model_list:
    data[model] = {'alpha':[],'count':[],'difference':[]}
    result_file = './normalize/{}/percent/prediction/10/score/{}/result_conclusion.csv'.format(model, score_type)
    pd_data = pd.read_csv(result_file)
    for _, item in pd_data.iterrows():
        alpha = item['map_function_scale']
        count = item['coherence count positive']
        diff = item['coherent mean improvement']
        if alpha != 'no':
            data[model]['alpha'].append(int(alpha))
            data[model]['count'].append(count)
            data[model]['difference'].append(diff)

color_list = ['b','r']
bar_width_list =[-0.2,0.2]
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
model_name_list = ['Bayesian Network','Linear Regression']
for index, model in enumerate(model_list):
    ax1.bar(x=np.array(data[model]['alpha'])+bar_width_list[index],height=data[model]['count'],label=model_name_list[index],color=color_list[index],width = 0.4,alpha=0.5)
    for x1, yy in zip(data[model]['alpha'], data[model]['count']):
        ax1.text(x1+bar_width_list[index], yy + 0.2, str(yy), ha='center', va='bottom', fontsize=20, rotation=0)
    ax2.plot(np.array(data[model]['alpha'])+bar_width_list[index], data[model]['difference'], color_list[index], label=model_name_list[index], linewidth=3)
ax1.set_ylim(0,14)
ax1.set_ylabel('# of improved metric', fontsize=20)
ax2.set_ylabel("Mean improvement", fontsize=20)
ax1.set_yticklabels(ax1.get_yticklabels(),size=20)
ax2.set_yticklabels(ax2.get_yticklabels(),size=20)
# ax2.yticks(size=20)
ax1.set_xlabel('Adjustment scale, in the scale of log 2',fontsize=20)
plt.xticks(size=20)
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0), columnspacing=0.7, handletextpad=0.3, ncol=2, prop={'size': 20})
plt.savefig('normalize-score.pdf', bbox_inches='tight', pad_inches=0)
