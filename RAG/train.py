# this file wraps RAG model and implements a simple training loop.

import torch
from rag_variants import *
from dataset import *
from tqdm import tqdm
from torch.utils.data import DataLoader

def train(model, dataset_loader, epochs=10):
    optim = torch.optim.AdamW(model.parameters(), lr=5e-4)
    for _ in (pbar := tqdm(range(epochs))):
        for contrastive_batch in dataset_loader:
            # source lang is bs x embedding_dim
            # target lang is bs x neg_samples x embedding_dim

            #embed
            embedded_source, embedded_target = model(contrastive_batch)

            def loss(embedded_source, embedded_target):
                """
                Implements InfoNCE loss
                """
                # compute the logits
                logits = torch.bmm(embedded_source.unsqueeze(1), embedded_target.transpose(1, 2))
                # softmax
                logits = torch.nn.functional.log_softmax(logits, dim=2).squeeze(1)

                # sum along the 0th column
                return -logits[:, 0].mean()

            loss_value = loss(embedded_source, embedded_target)
            loss_value.backward()
            optim.step()
            optim.zero_grad()
        pbar.set_description(f"loss: {loss_value}")

if __name__ == "__main__":

    inpt_embd_size = 300
    outpt_embd_size = 300
    index_dir = "rag-processed-datasets/index-vocab/en-es"
    embeddings_dir = "rag-processed-datasets/index-vocab/embeddings.json"



    # initialize model and dataset loader
    model = DecoupledRAG(index_dir, embeddings_dir)
    
    dataset_train = RAGDataset('rag-processed-datasets/en-de.csv')
    dataloader = DataLoader(dataset_train, batch_size=10, shuffle=False)

    # load the DPR model
    model.dpr.load_state_dict(torch.load('DPR_encoder.pt'))

    # compute a forward pass
    for data in dataloader:
        print(model(data))
        import sys
        sys.exit()
    with torch.no_grad():
        print(dataset_train[:10])
        print(model(dataset_train[:10]))

    #train(model, dataloader)

