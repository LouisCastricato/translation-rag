# this file loads a fasttext indexed dataset and wraps it in a pytorch dataset

import torch
from gensim.models.fasttext import load_facebook_vectors
from gensim.test.utils import datapath
from ..data_utils import *
import random

class FastTextDataset(torch.utils.data.Dataset):
    def __init__(self, dir):
        self.fasttext = load_facebook_vectors(datapath(dir))
        self.vocab_size = len(self.fasttext.key_to_index.keys())
        self.keys = list(self.fasttext.key_to_index.keys())

    def __len__(self):
        return self.vocab_size
    
    def get_random_key():
        """
        Returns a random word from the dataset
        """
        return self.keys[random.randint(0, len(self.keys))]
    def get_vector(self, word):
        """
        Returns a single word vector
        :param word: the word to get the vector for
        """
        return self.fasttext[word]
    
class PairedFastTextDataset(torch.utils.data.Dataset):
    def __init__(self, source_dir, target_dir, pairing_dir):
        self.source_dataset = FastTextDataset(source_dir)
        self.target_dataset = FastTextDataset(target_dir)

        # Given a set of vectors in the source dataset, find the closest vectors in the target dataset
        self.pair_dataset = load_space_delimited(pairing_dir)

    def __getitem__(self, index):
        """
        Constructs a batch
        :param index: The index of the gold tuple
        """
        source_word, target_word = self.pair_dataset[index]
        source_embedding = self.source_dataset.get_vector(source_word)
        target_embedding = self.target_dataset.get_vector(target_word)

        # use 100 random samples from the target dataset
        negative_samples = []
        for i in range(100):
            negative_samples.append(self.target_dataset.get_vector(self.target_dataset.get_random_key()))
        
        # use gold samples from other gold tuples
        


        
    def add_word(self, source_word):
        """
        Given the source word, finds a similar word in the target dataset
        :param source_word: the word to find a similar word for
        """
        source_vector = self.source_dataset.get_vector(source_word)


# test functionality
if __name__ == '__main__':
    dataset = FastTextDataset('/home/louis_huggingface_co/translation_rag/english/wiki.en.bin')
    print(dataset.vocab_size)
    print(dataset.get_vector('the'))