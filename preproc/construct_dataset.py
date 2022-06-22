import numpy as np 
from numpy import random
import sys

sys.path.append('.')
from data_utils import list_to_dict, load_space_delimited, save_dict

def slice_dict_rhs(dictionary : dict, length : int):
    return {k: v for k, v in list(dictionary.items())[:length]}
    
def slice_dict_lhs(dictionary : dict, length : int):
    return {k: v for k, v in list(dictionary.items())[length:]}


# dictionary where the key is an english word and the value is the word in a target language
en_de_dict = list_to_dict(load_space_delimited('data/dictionaries/en-de.train.txt'))
en_es_dict = list_to_dict(load_space_delimited('data/dictionaries/en-es.train.txt'))

# randomly select N key, value tuples from the dictionary
key_intersection = list(set(en_de_dict.keys()).intersection(set(en_es_dict.keys())))

# get our subset dictionary
en_de_dict = {k: en_de_dict[k] for k in  key_intersection}
en_es_dict = {k: en_es_dict[k] for k in  key_intersection}

# save the first 97.5% as the training set
en_es_training_length = int(len(en_es_dict)*0.60)
en_de_training_length = int(len(en_de_dict)*0.60)

save_dict(slice_dict_rhs(en_de_dict, en_de_training_length), 'DPR-input-data/en-de.train.csv')
save_dict(slice_dict_rhs(en_es_dict, en_es_training_length), 'DPR-input-data/en-es.train.csv')

# save the rest as the validation set
save_dict(slice_dict_lhs(en_de_dict, en_de_training_length), 'DPR-input-data/en-de.valid.csv')
save_dict(slice_dict_lhs(en_es_dict, en_es_training_length), 'DPR-input-data/en-es.valid.csv')