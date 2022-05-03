# this file implements bog standard contrastive learning for training a DPR model

import torch
from model import *
from dataset import *
from tqdm import tqdm

def train(model, dataset_loader):
    optim = torch.optim.AdamW(model.parameters(), lr=0.001)

    # the gold standard is always on the 0th index
    for contrastive_batch in tqdm(dataset_loader):
        # source lang is bs x embedding_dim
        # target lang is bs x neg_samples x embedding_dim

        #embed
        embedded_source, embedded_target = model.source_encoder(contrastive_batch)

        def loss(embedded_source, embedded_target):
            """
            Implements InfoNCE loss
            """
            # compute the logits
            logits = torch.bmm(embedded_source, embedded_target.transpose(1, 2))
            # softmax
            logits = torch.nn.functional.log_softmax(logits, dim=2)
            # sum along the 0th column
            return -torch.sum(logits, dim=1)

        loss_value = loss(embedded_source, embedded_target)
        loss.backward()
        optim.step()
        tqdm.write(f"loss: {loss_value}")
        optim.zero_grad()

if __name__ == "__main__":
    inpt_embd_size = 300
    outpt_embd_size = 300

    model = SourceTargetDPR(inpt_embd_size, outpt_embd_size)
    dataset = ContrastiveDataset(inpt_embd_size, outpt_embd_size)
