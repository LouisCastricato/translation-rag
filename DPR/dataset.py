# this file loads a fasttext indexed dataset and wraps it in a pytorch dataset
import sys
import torch
from gensim.models.fasttext import load_facebook_vectors
from gensim.test.utils import datapath
import random
from numpy.random import choice
import numpy as np

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
        positive = torch.tensor(self.target_dataset.get_vector(target_word))

        return {
            "anchor" : anchor.unsqueeze(0),
            "target" : positive.unsqueeze(0)
        }

# takes the dataset produced by PairedFastTextDataset, to avoid processing time
class ProcessedPairedTextDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir):
        self.data = load_json(dataset_dir)

        # convert the dataset that is saved as a list to torch tensors
        for k,v in self.data.items():
            self.data[k] = torch.tensor(v).cuda()

    def __len__(self):
        return self.data['anchor'].shape[0]
    
    def __getitem__(self, idx):
        return {
            "anchor" : self.data['anchor'][idx],
            "target" : self.data['target'][idx]
        }

class KnownDimensionalityDataset(torch.utils.data.Dataset):
    def __init__(self, suffix, base_dir = "../../known-intrinsic-dim/", is_val=False, val_split=0.9, seed=None):

        self.source = np.float32(np.load(base_dir + "datasets_A/A_" + str(suffix) + ".npy"))

        # generate randn in the shape of self.source
        #self.source = np.float32(np.random.randn(*self.source.shape))

        # generate a random rotation matrix to get the target
        # self.source is data_points x dim
        # rotation matrix is dim x dim
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        rotation_matrix = torch.nn.utils.parametrizations.orthogonal(torch.nn.Linear(self.source.shape[1], self.source.shape[1]))
        with torch.no_grad():
            self.target = rotation_matrix(torch.tensor(self.source)).numpy()

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
                                    '/home/louis_huggingface_co/translation_rag/spanish/wiki.es.bin',
                                    'DPR-input-data/en-es.train.csv')
    
    dataset_list = list()
    # save the dataset
    for i in range(len(dataset)):
        dataset_list.append(dataset[i])

    # each element of dataset_list is a dictionary, we need to concat the torch tensors
    dataset_dict = convert_dict_of_tensors_to_list(stack_dicts(dataset_list))
    save_to_json(dataset_dict, 'en-es.train.json')


    # save a copy of only the anchors, which we will use in the embedding layer
    for idx, elem in enumerate(dataset_list):
        dataset_list[idx] = {
            dataset.pair_dataset[idx][0] : elem['anchor'].numpy().tolist()}


    # each element of dataset_list is a dictionary, we need to concat the torch tensors
    dataset_dict = stack_dicts(dataset_list)
    save_to_json(dataset_dict, 'embeddings_en-es.train.json')

    dataset = PairedFastTextDataset('/home/louis_huggingface_co/translation_rag/english/wiki.en.bin',
                                    '/home/louis_huggingface_co/translation_rag/spanish/wiki.es.bin',
                                    'DPR-input-data/en-es.valid.csv')
    
    dataset_list = list()
    # save the dataset
    for i in range(len(dataset)):
        dataset_list.append(dataset[i])

    # each element of dataset_list is a dictionary, we need to concat the torch tensors
    dataset_dict = convert_dict_of_tensors_to_list(stack_dicts(dataset_list))
    save_to_json(dataset_dict, 'en-es.valid.json')


    # save a copy of only the anchors, which we will use in the embedding layer
    for idx, elem in enumerate(dataset_list):
        dataset_list[idx] = {
            dataset.pair_dataset[idx][0] : elem['anchor'].numpy().tolist()}
        
    # each element of dataset_list is a dictionary, we need to concat the torch tensors
    dataset_dict = stack_dicts(dataset_list)
    save_to_json(dataset_dict, 'embeddings_en-es.valid.json')

