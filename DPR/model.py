import sys
import torch
from torch import nn

sys.path.append('.')
from data_utils import load_json

# implements an MLP DPR model
class MultilayerDPR(torch.nn.Module):
    def __init__(self, input_size, output_size, dropout=0.1):
        """
        :param input_size: the size of the input embedding
        :param output_size: the size of the output embedding
        :param dropout: the dropout rate
        """
        super(MultilayerDPR, self).__init__()

        # Linear, ReLu, dropout
        self.l1 = nn.Linear(input_size, input_size//2)
        self.l2 = nn.Linear(input_size//2, output_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.input_size = input_size
        self.output_size = output_size
    
    def forward(self, x):
        """
        :param x: the input embedding. Of size bs x input_size
        :return: the output embedding. Of size bs x output_size
        """
        # apply the linear layer then relu
        return self.l2(self.relu(self.l1(x)))
    
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
        self.source_encoder = MultilayerDPR(input_size, output_size, dropout)
        self.target_encoder = MultilayerDPR(input_size, output_size, dropout)

    def embed_query(self, tok):
        """
        :param tok: the (single word) string to embed
        :return: the embedding of the string
        """

    def forward(self, x, duplicate=True, return_no_grad=False):
        """
        :param x: a tuple of batched source and target language embeddings
        :param duplicate: whether to duplicate the source embedding
        :param return_no_grad: whether to return a set of tensors with no grad
        :return: a tuple of batched source and target language embeddings, post projection
        """
        for k,v in x.items():
            x[k] = v.cuda()


        anchor = self.source_encoder(x["anchor"])
        target = self.target_encoder(x["target"])

        if return_no_grad:
            with torch.no_grad():
                anchor_no_grad = self.source_encoder(x["anchor"])
                target_no_grad = self.target_encoder(x["target"])

        # we need to duplicate the target
        if duplicate:
            target = torch.cat([target.unsqueeze(1)] * target.shape[0], dim=1).transpose(0,1)
            if return_no_grad:
                target_no_grad = torch.cat([target_no_grad.unsqueeze(1)] * target_no_grad.shape[0], dim=1).transpose(0,1)

        # normalize both anchor and target
        if not duplicate:
            anchor = torch.nn.functional.normalize(anchor, p=2, dim=1)
            target = torch.nn.functional.normalize(target, p=2, dim=1)
            if return_no_grad:
                anchor_no_grad = torch.nn.functional.normalize(anchor_no_grad, p=2, dim=1)
                target_no_grad = torch.nn.functional.normalize(target_no_grad, p=2, dim=1)

        if return_no_grad:
            return anchor, target, anchor_no_grad, target_no_grad
        else:
            return anchor, target