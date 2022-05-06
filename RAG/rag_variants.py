from model import BaseRAG
from typing import Iterable, Dict, Union, Tuple
from transformers.modeling_output import Seq2SeqLMOutput


# Decoupled RAG returns 0 for the coupling loss, since (duh) its decoupled.
class DecoupledRAG(BaseRAG):
    def __init__(self, index_dir : str = None, embd_size : int, dropout : float = 0.1):
        super(DecoupledRAG, self).__init__(index_dir, embd_size, dropout)

    def coupling_loss(self, x : Dict, model_output : Union[Tuple, Seq2SeqLMOutput], **kwargs):
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
        