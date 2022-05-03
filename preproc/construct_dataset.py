import numpy as np 
from numpy import random
import sys

sys.path.append('.')
from data_utils import *



# dictionary where the key is an english word and the value is the word in a target language
en_de_dict = list_to_dict(load_space_delimited('data/dictionaries/en-de.train.txt'))
en_es_dict = list_to_dict(load_space_delimited('data/dictionaries/en-es.train.txt'))

# randomly select N key, value tuples from the dictionary
key_intersection = list(set(en_de_dict.keys()).intersection(set(en_es_dict.keys())))

# get our subset dictionary
en_de_dict = {k: en_de_dict[k] for k in  key_intersection}
en_es_dict = {k: en_es_dict[k] for k in  key_intersection}


    
save_dict(en_de_dict, 'en-de.csv')
save_dict(en_es_dict, 'en-es.csv')

