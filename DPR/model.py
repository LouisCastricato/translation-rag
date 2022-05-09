import sys
import torch
from torch import nn

sys.path.append('.')
from data_utils import load_json

# implements a single layer DPR model
class SingeLayerDPR(torch.nn.Module):
    def __init__(self, input_size, output_size, dropout=0.1):
        """
        :param input_size: the size of the input embedding
        :param output_size: the size of the output embedding
        :param dropout: the dropout rate
        """
        super(SingeLayerDPR, self).__init__()

        # Linear, ReLu, dropout
        self.project = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.Dropout(dropout))
        
        self.input_size = input_size
        self.output_size = output_size
    
    def forward(self, x):
        """
        :param x: the input embedding. Of size bs x input_size
        :return: the output embedding. Of size bs x output_size
        """
        return self.project(x)
    
class EmbeddingLayer:
    def __init__(self, dataset_dir):
        """
        :param dataset_dir: the directory of the dictioanry containing the word, embedding tuples
        """
        # load the dictionary
        self.embeddings = load_json(dataset_dir)

        # convert the dataset that is saved as a list to torch tensors
        self.embeddings = {k: torch.tensor(v) for k, v in self.embeddings.items()}
    
    def __getitem__(self, word):
        """
        :param word: the word to get the embedding for
        :return: the embedding for the word
        """
        return self.embeddings[word]



class SourceTargetDPR(torch.nn.Module):
    def __init__(self, input_size, output_size, dropout=0.1):
        """
        :param input_size: the size of the input embedding
        :param output_size: the size of the output embedding
        :param dropout: the dropout rate
        """
        super(SourceTargetDPR, self).__init__()
        self.source_encoder = SingeLayerDPR(input_size, output_size, dropout)
        self.target_encoder = SingeLayerDPR(input_size, output_size, dropout)

    def embed_query(self, tok):
        """
        :param tok: the (single word) string to embed
        :return: the embedding of the string
        """

    def forward(self, x):
        """
        :param x: a tuple of batched source and target language embeddings
        :return: a tuple of batched source and target language embeddings, post projection
        """
        return self.source_encoder(x["anchor"]), self.target_encoder(x["target_batch"])