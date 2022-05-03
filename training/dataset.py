# this file loads a fasttext indexed dataset and wraps it in a pytorch dataset
import sys
import torch
from gensim.models.fasttext import load_facebook_vectors
from gensim.test.utils import datapath
import random
from numpy.random import choice

sys.path.append('.')
from data_utils import *



class FastTextDataset(torch.utils.data.Dataset):
    def __init__(self, dir, debug = False):
        self.fasttext = load_facebook_vectors(datapath(dir))
        self.vocab_size = len(self.fasttext.key_to_index.keys())
        self.keys = list(self.fasttext.key_to_index.keys())

    def __len__(self):
        return self.vocab_size
    
    def get_random_key(self):
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
        anchor = torch.tensor(self.source_dataset.get_vector(source_word))
        positive = self.target_dataset.get_vector(target_word)

        # use 100 random samples from the target dataset
        negative_samples = []
        for i in range(100):
            negative_samples.append(self.target_dataset.get_vector(self.target_dataset.get_random_key()))
        
        # use gold samples from other gold tuples
        # first randomly sample pair_dataset - index
        discrete_dist = list(range(len(self.pair_dataset)))
        discrete_dist.remove(index)

        sampled_gold = choice(discrete_dist, size=20)
        for idx in sampled_gold:
            negative_samples.append(self.target_dataset.get_vector(self.pair_dataset[idx][1]))

        # concat the target embedding and the negative samples
        positive = torch.tensor(positive).unsqueeze(1)
        negative_samples = list(map(lambda x: torch.tensor(x).unsqueeze(1), negative_samples))
        target_batch = torch.cat([positive] + negative_samples, dim=1)


        return {
            "anchor" : anchor.unsqueeze(0),
            "target_batch" : torch.transpose(target_batch).unsqueeze(0)
        }


# test functionality
if __name__ == '__main__':
    dataset = PairedFastTextDataset('/home/louis_huggingface_co/translation_rag/english/wiki.en.bin',
                                    '/home/louis_huggingface_co/translation_rag/spanish/wiki.es.bin',
                                    'en-es.csv')
    source, target = dataset[0]
    print(source.shape)
    print(target.shape)
