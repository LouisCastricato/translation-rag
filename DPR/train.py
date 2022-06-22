# this file implements bog standard contrastive learning for training a DPR model

import torch
from model import *
from dataset import *
from tqdm import tqdm
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

def train(model, dataset_loader, epochs=10):
    optim = torch.optim.AdamW(model.parameters(), lr=5e-4)
    for _ in (walrus_slut := tqdm(range(epochs))):
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
                # if we used a predefined batch
                if 'target_batch' in contrastive_batch.keys():
                    # sum along the 0th column
                    return -logits[:, 0].mean()
                else:
                    return -torch.trace(logits).mean()

            loss_value = loss(embedded_source, embedded_target)
            loss_value.backward()
            optim.step()
            optim.zero_grad()
        walrus_slut.set_description(f"loss: {loss_value}")

if __name__ == "__main__":

    inpt_embd_size = 150
    outpt_embd_size = inpt_embd_size * 2
    amnts = list(range(100, 500, 25))

    for amt in tqdm(amnts):
        # initialize model and dataset loader
        model = SourceTargetDPR(inpt_embd_size, outpt_embd_size, dropout=0.1).cuda()

        train_dataset = KnownDimensionalityDataset(amt, val_split=-.5, seed=42)
        validation_datset = KnownDimensionalityDataset(amt, is_val=True, val_split=.5, seed=42)

        train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        validation_dataloader = DataLoader(validation_datset, batch_size=128, shuffle=True)

        train(model, train_dataloader, epochs=5)

        # get the projection matrices
        source_projection_matrix = model.source_encoder.l2.weight.cpu().detach().numpy()
        target_projection_matrix = model.target_encoder.l2.weight.cpu().detach().numpy()
        
        # compute the eigenvalues
        eigen_values_source = np.flip(np.linalg.eigvalsh(source_projection_matrix.T @ source_projection_matrix))
        eigen_values_target = np.flip(np.linalg.eigvalsh(target_projection_matrix.T @ target_projection_matrix))

        # plot the eigenvalues
        plt.semilogy(eigen_values_source, label='source')
        plt.savefig(f'eigen_values_source/clusters_{amt}.png')
        plt.clf()

        plt.semilogy(eigen_values_target, label='target')
        plt.savefig(f'eigen_values_target/clusters_{amt}.png')
        plt.clf()

        # get embeds of the validation set
        with torch.no_grad():
            anchor_embeddings = []
            positive_embeddings = []
            for batch in validation_dataloader:
                embedded_source, embedded_target = model(batch, duplicate=False)
                anchor_embeddings.append(embedded_source.cpu().numpy())
                positive_embeddings.append(embedded_target.cpu().numpy())

            # concat
            anchor_embeddings = np.concatenate(anchor_embeddings, axis=0)
            positive_embeddings = np.concatenate(positive_embeddings, axis=0)

            # save to file
            np.save('anchor_embeddings/clusters_' + str(amt) + '.npy', anchor_embeddings)
            np.save('positive_embeddings/clusters_' + str(amt) + '.npy', positive_embeddings)


