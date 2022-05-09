import numpy as np
import sys
import torch
from torch.utils.data import DataLoader
from typing import List, Tuple, Iterable

from faiss_utils import DenseFlatIndexer
sys.path.append('.')
from data_utils import load_space_delimited
from DPR.dataset import ProcessedPairedTextDataset
from DPR.model import SourceTargetDPR

@torch.no_grad()
def embed(model : SourceTargetDPR, dataset : DataLoader, paired_words : Iterable) -> List[Tuple[str, np.array]]:
    """
    embeds the anchors and the first component of every target batch
    :param model: the model to embed with
    :param dataset: the dataset to embed with
    :return: A list of tuples where the first component is the corresponding word and the second component is the embedding
    """
    word_embedding = []
    for idx,contrastive_batch in enumerate(dataset):
        # source lang is bs x embedding_dim
        # target lang is bs x neg_samples x embedding_dim
        contrastive_batch["target_batch"] = contrastive_batch["target_batch"][:, 0]

        #embed
        _, embedded_target = model(contrastive_batch)
        word_embedding.append((paired_words[idx][1], embedded_target[0].detach().numpy()))
    return word_embedding



if __name__ == "__main__":
    model = SourceTargetDPR(300, 300, dropout=0.0)
    dataset = ProcessedPairedTextDataset('rag-processed-datasets/index-vocab/en-es.json')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    word_pairings = load_space_delimited('rag-processed-datasets/en-es.csv')

    model.load_state_dict(torch.load('DPR_encoder.pt'))

    embeddings = embed(model, dataloader, word_pairings)

    index = DenseFlatIndexer()
    # set the vector size
    index.init_index(300)
    # index the embeddings we just computed 
    index.index_data(embeddings)
    # save to disk
    index.serialize('rag-processed-datasets/index-vocab/en-es')
