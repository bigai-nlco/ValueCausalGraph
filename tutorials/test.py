# Standard imports
import os
import plotly.express as px
import pandas as pd
import json
import numpy as np
import math
import gc
import pandas as pd
import random
import shutil

from collections import Counter
from functools import partial
from tqdm import tqdm
from faker import Faker

# Imports for displaying vis in Colab / notebook
import webbrowser
import http.server
import socketserver
import threading
PORT = 8000

import torch
torch.set_grad_enabled(False);
from openai import AzureOpenAI
from datasets import load_dataset  
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

import transformer_lens
from transformer_lens import utils
from transformer_lens import HookedTransformer
from transformer_lens.utils import tokenize_and_concatenate

from sae_lens import SAE
from sae_lens.config import DTYPE_MAP, LOCAL_SAE_MODEL_PATH
from sae_lens.analysis.neuronpedia_integration import get_neuronpedia_quick_list

from sae_vis.data_config_classes import SaeVisConfig
from sae_vis.data_storing_fns import SaeVisData

from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ScoreBased.ExactSearch import bic_exact_search
from causallearn.utils.cit import kci
from causallearn.utils.GraphUtils import GraphUtils

def get_valid_d_columns_abondoned(answer_valuebench_features_csv):
    data_csv = pd.read_csv(answer_valuebench_features_csv)
    digits = [str(d) for d in range(10)]
    d_columns = [d for d in data_csv.columns if d[0] in digits]
    d_data = data_csv[d_columns]
    stds = d_data.std()
    avgs = d_data.mean()
    std_avg = stds/avgs
    #d_columns_valid = [d for d in d_columns if avgs[d] > 1]
    d_columns_valid = d_columns
    return d_columns_valid

def deal_with_csv(answer_valuebench_features_csv, pdy_name, v_inference, v_showongraph, row_num, method='pc'):
    data_csv = pd.read_csv(answer_valuebench_features_csv)
    v_columns_all = [v for v in data_csv.columns if (v not in ['player_name', 'steer_dim', 'stds']) and (not v.endswith(':scstd'))]
    
    if v_inference == 'ALL':
        v_columns_inference = v_columns_all
    else:
        for v in v_inference:
            if v not in v_columns_all:
                raise ValueError('Invalid v_inference')
        v_columns_inference = v_inference

    if v_showongraph == 'ALL':
        v_columns_showgraph = v_columns_inference
    else:
        for v in v_showongraph:
            if v not in v_columns_inference:
                raise ValueError('Invalid v_showongraph')
        v_columns_showgraph = v_showongraph

    data = data_csv[data_csv['player_name'].notnull()][v_columns_inference].to_numpy()    
    

    if type(row_num) == int:
        rows = np.random.choice(data.shape[0], row_num, replace=False)
        data = data[rows]
    else:
        assert row_num == 'ALL'
    causal_inference(data, v_columns_inference, pdy_name, method)
    return 
    #extract all data of a sinle steered dimension
    steer_dims = data_csv['steer_dim'].unique()
    for steer_dim in steer_dims:
        if np.isnan(steer_dim):
            data = data_csv[data_csv['steer_dim'].isnull()][v_columns_inference].to_numpy()
        else:
            data = data_csv[data_csv['steer_dim'] == steer_dim][v_columns_inference].to_numpy()
        print(steer_dim, 'begin')
        causal_inference(data, v_columns_inference, pdy_name.replace('.png', f'_{steer_dim}.png'), method)
        print(steer_dim, 'end')

def causal_inference(data, ci_dimensions, pdy_name, method):
    print(data.shape)
    
    #0 is the mean of the normal distribution you are choosing from, and 0.01 is the standard deviation of this distribution.
    noise = np.random.normal(0, 0.00001, data.shape)
    #data = data + noise


    if method == 'pc':
        #g = pc(data, 0.05, kci, kernelZ='Polynomial', node_names=ci_dimensions)
        g = pc(data, 0.0000005, node_names=ci_dimensions)
        graph = g.G
        edges = []
        for n1 in range(len(graph.nodes)):
            assert graph.nodes[n1].name == ci_dimensions[n1]
            for n2 in range(n1+1, len(graph.nodes)):
                # if n1 == n2:
                #     continue
                if graph.graph[n1][n2] == -1 and graph.graph[n2][n1] == 1:
                    edges.append([graph.nodes[n1].name, graph.nodes[n2].name, 1, 'single-arrow'])
                elif graph.graph[n1][n2] == 1 and graph.graph[n2][n1] == -1:
                    edges.append([graph.nodes[n2].name, graph.nodes[n1].name, 1, 'single-arrow']) 
                elif graph.graph[n1][n2] == -1 and graph.graph[n2][n1] == -1:
                    edges.append([graph.nodes[n1].name, graph.nodes[n2].name, 1, 'no-arrow'])
                elif graph.graph[n1][n2] == 1 and graph.graph[n2][n1] == 1:
                    edges.append([graph.nodes[n1].name, graph.nodes[n2].name, 1, 'double-arrow'])
                else:
                    if not (graph.graph[n1][n2] == 0 and graph.graph[n2][n1] == 0):
                        raise ValueError('Invalid edge')
                    
    
    elif method == 'fci':
        graph, edges = fci(data)
        pass
        #g, fciedges = fci(data)
        # or customized parameters
        #g, edges = fci(data, independence_test_method, alpha, depth, max_path_length,
        #    verbose, background_knowledge, cache_variables_map)
    else:
        raise ValueError('Invalid method')


    columns_concerned_vis = [label.replace(':','-') for label in ci_dimensions]
    pdy = GraphUtils.to_pydot(graph, labels=columns_concerned_vis)
    pdy.write_png(pdy_name)
    return edges

#v_inference = 'ALL'

#v_inference = ['Affiliation', 'Assertiveness', 'Behavioral Inhibition System', 'Breadth of Interest', 'Complexity', 'Dependence', 'Depth', 'Emotional Expression', 'Emotional Processing', 'Empathy', 'Extraversion', 'Imagination', 'Nurturance', 'Perspective Taking', 'Poise', 'Positive Expressivity', 'Preference for Order and Structure', 'Privacy', 'Psychosocial flourishing', 'Reflection']

v_inference = ['Affiliation', 'Assertiveness', 'Behavioral Inhibition System', 'Breadth of Interest', 'Complexity', 'Dependence', 'Depth', 'Emotional Expression', 'Emotional Processing', 'Empathy', 'Extraversion', ]


#############################################
answer_valuebench_features_csv = "ans_cross_part1.csv"
edges1 = deal_with_csv(answer_valuebench_features_csv, "value_causal_graph/total1.png", v_inference, 'ALL', 'ALL', 'pc')

answer_valuebench_features_csv = "ans_cross_part2.csv"
edges1 = deal_with_csv(answer_valuebench_features_csv, "value_causal_graph/total2.png", v_inference, 'ALL', 'ALL', 'pc')

steer_dims = ['nan', 1312, 1341, 2221, 3183, 6619, 7502, 8387, 10096, 14049]
for steer_dim in steer_dims:
    answer_valuebench_features_csv = "ans_steer_" + str(steer_dim) + ".csv"
    edges1 = deal_with_csv(answer_valuebench_features_csv, "value_causal_graph/" + str(steer_dim) + ".png", v_inference, 'ALL', 'ALL', 'pc')

assert False


nodes = {}
for entity in smallset:
    nodes[entity] = os.path.join('valuebench','value_questions_' + entity + '.html'),
for feature in d_columns_valid:
    nodes[feature] = 'https://www.neuronpedia.org/' + sae.cfg.model_name +'/' + str(sae.cfg.hook_layer) + '-res-jb/' + str(feature)



edges = {
    'gemma2bit-smallset-collect': edges1,
    # 'human_annotated': edges0,
    # 'gemma2bit-unsure-smallset-30': edges1
    #'llama38bit_cismall_30_trial1': edges1,
    #'llama38bit_cismall_30_trial2': edges11,
    #'llama38bit_cismall_250_trial1': edges2,
    #'llama38bit_cismall_250_trial2': edges21,
    #'llama38bit_ciall4small_250': edges3,
    #'llama38bit_cismall_250_lessqa': edges4,
    #'llama38bit_cismall_250_chn': edges5,
}

json_object = {
    'nodes': nodes,
    'edges': edges
    }

#json.dump(json_object, open('data1.json', 'w'))


