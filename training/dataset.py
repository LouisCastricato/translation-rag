# this file loads a fasttext indexed dataset and wraps it in a pytorch dataset

import torch
from gensim.models.fasttext import load_facebook_vectors
from gensim.test.utils import datapath

class FastTextDataset(torch.utils.data.Dataset):
    def __init__(self, dir):
        self.fasttext = load_facebook_vectors(datapath(dir))
        self.vocab_size = len(self.fasttext.key_to_index.keys())
    
    def __len__(self):
        return self.vocab_size
    
    def get_vector(self, word):
        """
        Returns a single word vector
        :param word: the word to get the vector for
        """
        return self.fasttext[word]
    
class PairedFastTextDataset(torch.utils.data.Dataset):
    def __init__(self, source_dir, target_dir):
        self.source_dataset = FastTextDataset(source_dir)
        self.target_dataset = FastTextDataset(target_dir)

        # a dictionary of the form {source_word: target_word}
        self.word_pairs = {}

        # Given a set of vectors in the source dataset, find the closest vectors in the target dataset
        self.pair_dataset = {}

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