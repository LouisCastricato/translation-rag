# implements a very basic RAG model for experimentation

import torch
import sys
from typing import Iterable, Dict, Union, Tuple
from transformers import BartTokenizer, BartConfig, BartForConditionalGeneration
from transformers.modeling_output import Seq2SeqLMOutput

sys.path.append('.')
from DPR.model import SourceTargetDPR

class BaseRAG(torch.nn.Module):
    def __init__(self, index_dir : str = None, embd_size : int, dropout : float = 0.1):
        super(BaseRAG, self).__init__()
        """
        :param index_dir: the directory of the FAISS index + embedding dictionary
        :param embd_size: the size of the embedding
        :param dropout: the dropout rate
        """
        self.index_dir = index_dir
        self.embedding_dir = index_dir + '/embeddings.json'
        self.embedding_layer = EmbeddingLayer(embedding_dir)
        print("Embedding layer loaded.")

        # initialize faiss index
        self.faiss = DenseFlatIndexer()
        self.fass.init_index(300)
        self.faiss.deserialize(index_dir)
        print("FAISS index loaded.")

        # initalize DPR
        self.dpr = SourceTargetDPR(embd_size, embd_size, dropout)

        # initalize the reader model
        self.lm_config = BartConfig(
            encoder_layers=2, 
            decoder_layers=2, 
            d_model=embd_size, 
            dropout=dropout)

        self.language_model = GPT2LMHeadModel(lm_config)
        self.tokenizer = BartTokenizer.from_pretrained('bart-large')
        print("Model initialized.")

        self.embd_size = embd_size
    
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
        raise NotImplementedError

    def query(self, x : Dict, k : int = 1):
        """
        Queries against a FAISS index by first embedding query and then querying the index
        :param x: a dictionary containing atleast the following keys:
        {'query_word': sequence of source words}
        :param k: the number of results to return
        :return: a dictionary containing the following keys:
        {'source_word': list of length k of source word,
        'query_embedding': query embedding}
        """
        # embed the string and then project
        query_embedding = self.dpr.source_encoder(self.embedding_layer[x["query_word"]])

        # get top-k results for our query.
        search_results = self.faiss.search_knn(query_embedding.detach().cpu().numpy(), k)

        return {
            'source_word_id': search_results,
            'query_embedding': query_embedding
        }



    def tokenize(self, sequence : Iterable[str], max_length : int = 5):
        """
        Tokenizes a sequence of strings
        :param sequence: the sequence to tokenize
        :param max_length: the maximum length of the sequence
        :return: a tensor of tokenized sequences
        """
        return self.tokenizer.batch_encode_plus(
                    sequence,
                    pad_to_max_length=True, 
                    max_length=5,
                    truncation=True,
                    return_tensors="pt") # ~ bs x 5
        
    def forward(self, x : Dict):
        """
        Forward pass of the model
        :param x: a dictionary containing atleast the following keys:
        {'query_word': sequence of query words, 
        'target_word': sequence of target words}
        :return: generative_loss + coupling loss
        """
        inpt_word = self.tokenize(x["query_word"]) # ~ bs x 5

        # run the query
        query_results = self.query(x)

        # get the source ids and embeddings for each query.
        source_id = query_results['source_word_id']
        query_embedding = query_results['query_embedding']

        # get our labels
        target_word = self.tokenize(x["target_word"]) # ~ bs x 5


        # run a forward pass on BART, record loss
        model_output = self.language_model(
            input_ids=inpt_word.input_ids,
            attention_mask=inpt_word.attention_mask, 
            labels=target_word.input_ids)

        # compute RAG's coupling loss
        coupling_loss = self.coupling_loss(x, model_output)

        return model_output.loss + coupling_loss
