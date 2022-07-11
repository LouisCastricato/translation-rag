# this file loads the result of the FAISS indexing and uses it construct a FAISS dataset
import sys
import torch
from gensim.models.fasttext import load_facebook_vectors
from gensim.test.utils import datapath
import random
from numpy.random import choice

sys.path.append('.')
from data_utils import *
from DPR.dataset import *
from indexing.faiss_utils import DenseFlatIndexer

class RAGDataset(torch.utils.data.Dataset):
    def __init__(self, pairing_directory):
        self.pair_dataset = load_space_delimited(pairing_directory)

    def __len__(self):
        """
        Returns the number of pairs
        :return: number of pairs
        """
        return len(self.pair_dataset)

    def __getitem__(self, idx):
        """
        Returns a pair of (query, target)
        :param idx: index of the pair
        :return: dictionary containing the keys:
        {'query_word': the word to query with
        'target_word': the target translation word}
        """
        return {
            'query_word': self.pair_dataset[idx][0],
            'target_word': self.pair_dataset[idx][1],
            'idx': idx
        }
