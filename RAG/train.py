# this file wraps RAG model and implements a simple training loop.

import torch
from rag_variants import *
from dataset import *
from tqdm import tqdm
from torch.utils.data import DataLoader

def train(model, dataset_loader, epochs=10):
    optim = torch.optim.AdamW(model.parameters(), lr=5e-4)
    for _ in (pbar := tqdm(range(epochs))):
        for batch in dataset_loader:
            
            # forward pass
            model_output =  model(batch)

            # loss and backwards step
            loss_value = model_output['Loss/Total']
            loss_value.backward()
            optim.step()
            optim.zero_grad()

        # update tqdm bar every epoch with the loss
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

    # train the model
    train(model, dataloader)

    # save the model
    torch.save(model.state_dict(), 'RAG.pt')

