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
        embedded_source, _ = model(contrastive_batch, duplicate=False)
        word_embedding.append((paired_words[idx][1], embedded_source.cpu().numpy()))
    return word_embedding

def save_index(embeddings, suffix="train"):
    index = DenseFlatIndexer()
    index.init_index(300)
    index.index_data(embeddings)
    index.serialize('rag-processed-datasets/index-vocab/en-de.' + suffix)

if __name__ == "__main__":
    model = SourceTargetDPR(300, 300, dropout=0.0).cuda()
    dataset = ProcessedPairedTextDataset('DPR-processed-data/en-de.train.json')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    word_pairings = load_space_delimited('rag-processed-datasets/en-de.csv')

    model.load_state_dict(torch.load('DPR_encoder.pt'))
    embeds = embed(model, dataloader, word_pairings)
    
    train_embeds = embeds[:int(len(embeds) * 0.9)]
    valid_embeds = embeds[int(len(embeds) * 0.9):]

    save_index(train_embeds, suffix="train")
    save_index(valid_embeds, suffix="valid")
