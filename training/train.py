# this file implements bog standard contrastive learning for training a DPR model

import torch
from model import *
from dataset import *
from tqdm import tqdm
from torch.utils.data import DataLoader

def train(model, dataset_loader):
    optim = torch.optim.AdamW(model.parameters(), lr=0.001)
    bs = 10
    neg_samples = 100
    embd_dim = 300 

    for contrastive_batch in tqdm(dataset_loader):
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
        tqdm.write(f"loss: {loss_value}")
        optim.zero_grad()

if __name__ == "__main__":

    inpt_embd_size = 300
    outpt_embd_size = 300

    # initialize model and dataset loader
    model = SourceTargetDPR(inpt_embd_size, outpt_embd_size)
    dataset = PairedFastTextDataset('/home/louis_huggingface_co/translation_rag/english/wiki.en.bin',
                                    '/home/louis_huggingface_co/translation_rag/spanish/wiki.es.bin',
                                    'en-es.csv')
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    train(model, dataloader)

    # save the model
    torch.save(model.state_dict(), 'model.pt')
