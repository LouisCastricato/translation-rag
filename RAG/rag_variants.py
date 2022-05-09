from model import BaseRAG
from typing import Iterable, Dict, Union, Tuple, Any


# Decoupled RAG returns 0 for the coupling loss, since (duh) its decoupled.
class DecoupledRAG(BaseRAG):
    def __init__(
        self, 
        index_dir : str = None, 
        embedding_dir = None):

        super(DecoupledRAG, self).__init__(index_dir, embedding_dir)

    def coupling_loss(self, x : Dict, model_output : Union[Tuple, Any], **kwargs):
        """
        Computes the coupling loss for the query embedding model
        :param x: a dictionary containing atleast the following keys:
        {'query_word': sequence of query words,
        'source_word': sequence of source words,
        'target_word': sequence of target words}
        :param model_output: the output of the language model
        :return: the loss
        """
        return 0
        