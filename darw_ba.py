from pomegranate import BayesianNetwork
import numpy as np
import pandas as pd
from typing import List, Tuple
import random
import os
import json
import tqdm
from multiprocessing import Pool
from collections import defaultdict
import sys
import matplotlib.pyplot as plt


key_list = ['source article length',
                'reference length', 'generated summary length', 'score']
data = [[1,1,1,1],[1,1,2,1]]
edge_list = [(0,2),(0,3),(1,3),(2,3)]
exclude_edges = [(0,1),(1,0),(1,2),(2,0),(2,1)]

ba_model = BayesianNetwork.from_samples(
        data,state_names = key_list,include_edges=edge_list, exclude_edges=exclude_edges)

ba_model.plot('./analyze_model.pdf')


key_list = ['model','source article length',
                'reference length', 'generated summary length', 'score']
data = [[1,1,1,1,1],[1,1,1,2,1]]
edge_list = [(0,3),(0,4),(1,3),(1,4),(2,4),(3,4)]
exclude_edges = [(0,1),(0,2),(1,0),(1,2),(2,0),(2,1),(2,3),(3,0),(3,1),(3,2)]


ba_model = BayesianNetwork.from_samples(
        data,state_names = key_list,include_edges=edge_list, exclude_edges=exclude_edges)

ba_model.plot('./normalize_model.pdf')