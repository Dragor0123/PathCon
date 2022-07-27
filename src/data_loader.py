import os
import re
import pickle
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from utils import count_all_paths_with_mp, count_paths, get_path_dict_and_length, one_hot_path_id, sample_paths


def load_data(model_args):
    global args, entity_dict, relation_dict
    args = model_args
    directory = '../data/' + args.dataset + '/'

    # print('reading entity dict and relation dict ...')
    entity_dict = read_entities(directory + 'entities.dict')
    relation_dict = read_relations(directory + 'relations.dict')
