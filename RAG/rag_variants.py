from model import BaseRAG
import torch
import torch.nn.functional as F
from typing import Iterable, Dict, Union, Tuple, Any

# Decoupled RAG returns 0 for the coupling loss, since (duh) its decoupled.
class DecoupledRAG(BaseRAG):
    def __init__(
        self, 
        index_dir : str = None, 
        embedding_dir = None):

        super(DecoupledRAG, self).__init__(index_dir, embedding_dir)

    def coupling_loss(self, 
        x : Dict, 
        model_output : Union[Tuple, Any], 
        query_embedding : torch.tensor,
        source_embedding : torch.tensor, 
        **kwargs):
        """
        Computes the coupling loss for the query embedding model
        :param x: a dictionary containing atleast the following keys:
        {'query_word': sequence of query words,
        'source_word': sequence of source words,
        'target_word': sequence of target words}
        :param model_output: the output of the language model
        :param query_embedding: the query embedding
        :param source_embedding: the source document embedding
        :return: the loss
        """
        return 0

# This is the standard RAG implementation that uses a marginalized NLL loss.
class MarginalizedRAG(BaseRAG):
    def __init__(
        self, 
        index_dir : str = None, 
        embedding_dir = None):

        super(MarginalizedRAG, self).__init__(index_dir, embedding_dir)

    def coupling_loss(self, 
        x : Dict, 
        model_output : Union[Tuple, Any], 
        query_embedding : torch.tensor, 
        source_embedding : torch.tensor,
        **kwargs):
        """
        Computes the coupling loss for the query embedding model
        :param x: a dictionary containing atleast the following keys:
        {'query_word': sequence of query words,
        'source_word': sequence of source words,
        'target_word': sequence of target words}
        :param model_output: the output of the language model
        :param query_embedding: the query embedding
        :param source_embedding: the source document embedding
        :return: the loss
        """
        # get the number of documents 
        bs, k, embd_size = query_embedding.shape

        # resize query and source embedding to (-1, embd_size)
        query_embedding = query_embedding.view(-1, embd_size)
        source_embedding = source_embedding.view(-1, embd_size)

        # generate the indices 
        indices = torch.arange(k).to(query_embedding)
        # stack the indices by the bs 
        indices = indices.unsqueeze(0).repeat(bs, 1)
        # flatten the indices and cast to long
        indices = indices.view(-1).long()

        # compute the marginal distribution 
        margin_dist = F.softmax(query_embedding @ source_embedding.t(), dim=1)

        # retrieve the corresponding logits
        logits = torch.zeros((bs * k))
        for i, idx in enumerate(indices):
            logits[idx] = margin_dist[i, idx]

        #logits = margin_dist[:, indices]
        print(logits)

        print(query_embedding.shape)
        print(source_embedding.shape)

        print(margin_dist.shape)
        return 0

