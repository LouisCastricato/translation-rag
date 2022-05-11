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
        labels : torch.tensor,
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
        :param labels: the labels
        :return: the loss
        """
        return 0

# This is the standard RAG implementation that uses a marginalized NLL loss.
class SequenceMarginalizedRAG(BaseRAG):
    def __init__(
        self, 
        index_dir : str = None, 
        embedding_dir = None):

        super(SequenceMarginalizedRAG, self).__init__(index_dir, embedding_dir)

    def coupling_loss(self, 
        x : Dict, 
        model_output : Union[Tuple, Any], 
        query_embedding : torch.tensor, 
        source_embedding : torch.tensor,
        labels : torch.tensor,
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
        :param ar_loss: the non-reduced autoregressive loss
        :return: the loss
        """
        # get the number of documents 
        bs, k, embd_size = query_embedding.shape

        # resize query and source embedding to (-1, embd_size)
        query_embedding = query_embedding.view(-1, embd_size).cuda()
        source_embedding = source_embedding.view(-1, embd_size).cuda()

        # generate the indices 
        indices = torch.arange(k).to(query_embedding)
        # stack the indices by the bs 
        indices = indices.unsqueeze(0).repeat(bs, 1)
        # flatten the indices and cast to long
        indices = indices.view(-1).long()

        # compute the marginal distribution 
        margin_dist = query_embedding @ source_embedding.t()

        # retrieve the corresponding logits
        # TODO: Verify that this is correct + speed up
        doc_logits = torch.zeros((bs * k)).cuda()
        for i, idx in enumerate(indices):
            doc_logits[i] = margin_dist[i, idx] 

        doc_logprobs = F.log_softmax(doc_logits.view(bs, k), dim=-1).view(-1)

        # combine the autoregressive logprobs and the doc_logprobs
        # we only want to utilize doc logits on the second token
        ar_logprobs = F.log_softmax(model_output.logits, dim=-1)

        first_token_scores = ar_logprobs[:, 0, :].unsqueeze(1)
        second_token_scores = ar_logprobs[:, 1, :]
        remainder = ar_logprobs[:, 2:, :]


        
        second_token_scores = (second_token_scores + doc_logprobs.unsqueeze(1)).unsqueeze(1)
        rag_logprobs = torch.cat([first_token_scores, second_token_scores, remainder], dim=1)

        return self.autoregressive_loss(rag_logprobs, labels, reduction='mean')
