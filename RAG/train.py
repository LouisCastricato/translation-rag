# this file wraps RAG model and implements a simple training loop.

import torch
from rag_variants import *
from dataset import *
from tqdm import tqdm
from torch.utils.data import DataLoader

import sys
sys.path.append('../')
from DPR.train_utils import embed_valid

def train(model, dataset_loader, valid_loader, epochs=10, valid_every=10):
    count = 0
    optim = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    for _ in (pbar := tqdm(range(epochs))):
        for batch in dataset_loader:
            
            # forward pass
            model_output =  model(batch)

            # loss and backwards step
            loss_value = model_output['Loss']
            loss_value.backward()
            optim.step()
            optim.zero_grad()
            if count % valid_every == 0:
                model.eval()
                with torch.no_grad():
                    embed_valid(model.dpr, valid_loader, "RAG_embeddings_" + str(count))
                model.train()
            count += 1

        # update tqdm bar every epoch with the loss
        pbar.set_description(f"loss: {loss_value}")

if __name__ == "__main__":

    inpt_embd_size = 300
    outpt_embd_size = 300
    index_dir = "rag-processed-datasets/index-vocab/en-es"
    embeddings_dir = "DPR-processed-data/embeddings_en-es.json"

    valid_dataset = ProcessedPairedTextDataset('DPR-processed-data/en-es.train.json')
    valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=True)

    dataset_train = RAGDataset('DPR-input-data/en-de.train.csv')
    dataloader = DataLoader(dataset_train, batch_size=100, shuffle=True)

    # initialize model and dataset loader
    model = SequenceMarginalizedRAG(index_dir, embeddings_dir, vocab_size=len(dataset_train)).cuda()
    
    # load the DPR model
    model.dpr.load_state_dict(torch.load('DPR_encoder.pt'))

    # train the model
    train(model, dataloader, valid_loader, epochs=20)

    # save the model
    torch.save(model.state_dict(), 'RAG.pt')