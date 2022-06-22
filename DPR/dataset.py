# this file loads a fasttext indexed dataset and wraps it in a pytorch dataset
import sys
import torch
from gensim.models.fasttext import load_facebook_vectors
from gensim.test.utils import datapath
import random
from numpy.random import choice
import numpy as np
from modalcollapse.dataset_generation import generate_a_random_rotation_matrix

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
        return self.keys[random.randint(0, len(self.keys) - 1)]
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

    def __len__(self):
        return len(self.pair_dataset)
        
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
            "target_batch" : torch.transpose(target_batch, 0, 1).unsqueeze(0)
        }

# takes the dataset produced by PairedFastTextDataset, to avoid processing time
class ProcessedPairedTextDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir):
        self.data = load_json(dataset_dir)

        # convert the dataset that is saved as a list to torch tensors
        for k,v in self.data.items():
            self.data[k] = torch.tensor(v)

    def __len__(self):
        return self.data['anchor'].shape[0]
    
    def __getitem__(self, idx):
        return {
            "anchor" : self.data['anchor'][idx],
            "target_batch" : self.data['target_batch'][idx]
        }

class KnownDimensionalityDataset(torch.utils.data.Dataset):
    def __init__(self, suffix, base_dir = "../../known-intrinsic-dim/", is_val=False, val_split=0.9, seed=None):

        self.source = np.float64(np.load(base_dir + "datasets_A/A_" + str(suffix) + ".npy"))

        # generate a random rotation matrix to get the target
        # self.source is data_points x dim
        # rotation matrix is dim x dim
        if seed is not None:
            np.random.seed(seed)
        rotation_matrix = generate_a_random_rotation_matrix(dim=self.source.shape[1])
        self.target = np.matmul(self.source, rotation_matrix)

        #convert to float32
        self.source = np.float32(self.source)
        self.target = np.float32(self.target)

        if not is_val:
            self.source = self.source[:int(val_split * len(self.source))]
            self.target = self.target[:int(val_split * len(self.target))]
        else:
            self.source = self.source[int(val_split * len(self.source)):]
            self.target = self.target[int(val_split * len(self.target)):]

    def __len__(self):
        return len(self.source)
    
    def __getitem__(self, idx):
        return {
            "anchor" : torch.tensor(self.source[idx]),
            "target" : torch.tensor(self.target[idx])
        }

# test functionality
if __name__ == '__main__':
    dataset = PairedFastTextDataset('/home/louis_huggingface_co/translation_rag/english/wiki.en.bin',
                                    '/home/louis_huggingface_co/translation_rag/german/wiki.de.bin',
                                    'DPR-input-data/en-de.valid.csv')
    
    dataset_list = list()
    # save the dataset
    for i in range(len(dataset)):
        dataset_list.append(dataset[i])

    # each element of dataset_list is a dictionary, we need to concat the torch tensors
    dataset_dict = convert_dict_of_tensors_to_list(stack_dicts(dataset_list))
    save_to_json(dataset_dict, 'en-de.valid.json')


    # save a copy of only the anchors, which we will use in the embedding layer
    for idx, elem in enumerate(dataset_list):
        dataset_list[idx] = {
            dataset.pair_dataset[idx][0] : elem['anchor'].numpy().tolist()}
        
    # each element of dataset_list is a dictionary, we need to concat the torch tensors
    dataset_dict = stack_dicts(dataset_list)
    #save_to_json(dataset_dict, 'embeddings.json')

