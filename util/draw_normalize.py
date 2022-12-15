import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl


model_list = ['bayesian','linear']
score_type = 'kendallatu'

data = dict()
for model in model_list:
    data[model] = {'alpha':[],'positive_count':[],'tier_count':[], 'negative_count':[],'difference':[], 'min_difference':[],'max_difference':[]}
    # result_file = './normalize/{}/percent/prediction/10/score/{}/result_conclusion.csv'.format(model, score_type)
    result_file = './normalize/{}/percent/prediction/10/score/{}/result_coherency.csv'.format(model, score_type)
    pd_data = pd.read_csv(result_file)
    for _, item in pd_data.iterrows():
        item = item.to_dict()
        alpha = item['map_function_scale']
        positive_count = item['positive count']
        tier_count = item['tier count']
        negative_count = item['negative count']
        min_diff = item['min improvement']
        max_diff = item['max improvement']
        diff = item['mean improvement']
        if alpha != 'no':
            data[model]['alpha'].append(int(alpha))
            data[model]['positive_count'].append(positive_count)
            data[model]['tier_count'].append(tier_count)
            data[model]['negative_count'].append(negative_count)

            data[model]['min_difference'].append(min_diff)
            data[model]['max_difference'].append(max_diff)
            data[model]['difference'].append(diff)

# color_list = ['C0','C1']
color_list = ['b','r']
marker_list = ['*','x']
bar_width_list =[-0.2,0.2]
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
model_name_list = ['BN','LR']
handles_list = list()
for index, model in enumerate(model_list):

    handles = ax1.bar(x=np.array(data[model]['alpha'])+bar_width_list[index],height=data[model]['positive_count'],label='{} win'.format(model_name_list[index]),color=plt.cm.tab20c(1+index*4),width = 0.4,alpha=0.5)
    handles_list.append(handles)
    
    handles = ax1.bar(x=np.array(data[model]['alpha'])+bar_width_list[index],height=data[model]['tier_count'],bottom=data[model]['positive_count'], label='{} Tier'.format(model_name_list[index]),color=plt.cm.tab20c(2+index*4),width = 0.4,alpha=0.5,hatch="x")
    handles_list.append(handles)

    bottom_list = data[model]['positive_count']
    for j, h in enumerate(data[model]['tier_count']):
        bottom_list[j] += h

    handles = ax1.bar(x=np.array(data[model]['alpha'])+bar_width_list[index],height=data[model]['negative_count'],bottom=bottom_list, label='{} Lose'.format(model_name_list[index]),color=plt.cm.tab20c(3+index*4),width = 0.4,alpha=0.5,hatch="*")
    handles_list.append(handles)

    handles, = ax2.plot(np.array(data[model]['alpha']), data[model]['difference'], color=plt.cm.tab20c(0+index*4), label=model_name_list[index], linewidth=3)
    # ax2.fill_between(np.array(data[model]['alpha']),data[model]['max_difference'],data[model]['min_difference'], alpha=0.5)
    handles_list.append(handles)


ax1.set_ylim(0,14)
ax1.set_ylabel('# of metrics', fontsize=20)
ax2.set_ylabel("Mean improvement", fontsize=20)
ax1.set_yticklabels(ax1.get_yticklabels(),size=20)
ax2.set_yticklabels(ax2.get_yticklabels(),size=20)
# ax2.yticks(size=20)
ax1.set_xlabel('Adjustment scale, in the scale of log 2',fontsize=20)
plt.xticks(size=20)
# fig.legend(handles=handles_list, loc='upper center', bbox_to_anchor=(0.5, 0), columnspacing=0.7, handletextpad=0.3, ncol=4, prop={'size': 20})
fig.legend(handles=handles_list, loc='upper center', bbox_to_anchor=(0.5, 0), columnspacing=0.7, handletextpad=0.3, ncol=4, prop={'size': 20})
plt.savefig('./paper/normalize-score.pdf', bbox_inches='tight', pad_inches=0)
