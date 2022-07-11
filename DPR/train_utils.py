import numpy as np
import torch
from tqdm import tqdm


def InfoNCE(embedded_source, embedded_target):
    """
    Implements InfoNCE loss
    """
    logits = torch.bmm(embedded_source.unsqueeze(1), embedded_target.transpose(1, 2))
    # softmax
    logits_i = torch.nn.functional.log_softmax(logits, dim=-1).squeeze()
    logits_j = torch.nn.functional.log_softmax(logits, dim=-2).squeeze()

    return -logits_i.diag().mean() + logits_j.diag().mean()

def validate(model, valid_dataloader):
    """
    Compute the validation loss
    :param model: DPR model
    :param valid_dataloader: dataloader for the validation set
    :return: validation loss
    """
    with torch.no_grad():
        loss_value = 0
        for batch in valid_dataloader:
            embedded_source, embedded_target = model(batch)
            loss_value += InfoNCE(embedded_source, embedded_target)
    return loss_value / len(valid_dataloader)


def train(model, dataset_loader, validation_loader=None, epochs : int = 10, noise = None, embed : int = None):
    """
    Train the model
    :param model: DPR model
    :param dataset_loader: dataloader for the training set
    :param epochs: number of epochs to train
    :param noise: noise to add to the source embedder
    :param embed: whether to embed the validation set. Integer
    """
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    embed_count = 0
    for _ in (pbar := tqdm(range(epochs))):
        model.train()
        for idx, contrastive_batch in enumerate(dataset_loader):
            # source lang is bs x embedding_dim
            # target lang is bs x embedding_dim

            #embed
            if noise is not None:
                embedded_source, embedded_target, embedded_source_no_grad, embedded_target_no_grad =\
                    model(contrastive_batch, return_no_grad=True)

                embedded_source_no_grad += torch.randn_like(embedded_source) * noise[embed_count]
                embedded_target_no_grad += torch.randn_like(embedded_source) * noise[embed_count]

                # loss
                loss_1 = InfoNCE(embedded_source_no_grad, embedded_target)
                loss_2 = InfoNCE(embedded_source, embedded_target_no_grad)
                loss_value = (loss_1 + loss_2)/2

            else:
                embedded_source, embedded_target = model(contrastive_batch)

                # loss
                loss_value = InfoNCE(embedded_source, embedded_target)

            # backwards
            loss_value.backward()

            # optimizer
            optim.step()
            optim.zero_grad()

            # if embed is not none, embed the validation set every embed steps
            if embed is not None:
                if idx % embed == 0:
                    with torch.no_grad():
                        embed_valid(model, dataset_loader, name = str(embed_count),\
                            anchor_root='time_anchor_embeddings', positive_root='time_positive_embeddings')
                        embed_count += 1
                    

        # if we provide a validation set, compute the validation loss and report
        if validation_loader is not None:
            model.eval()
            validation_loss = validate(model, validation_loader)
            pbar.set_description(f"Train loss: {loss_value}, Validation loss: {validation_loss}")
        else:
            pbar.set_description(f"Train loss: {loss_value}")
    return model

def embed_valid(model, valid_dataloader, 
    name : str, anchor_root : str = 'anchor_embeddings', 
    positive_root : str = 'positive_embeddings', skip_positive : bool = False):
    """
    Returns embeddings of the validation set
    :param model: DPR model
    :param valid_dataloader: dataloader for the validation set
    :param name: identifier for anchor and positive embeddings
    :param anchor_root: root for anchor embeddings (file directory)
    :param positive_root: root for positive embeddings (file directory)
    :param skip_positive: whether to skip the positive embeddings
    """
    with torch.no_grad():
        anchor_embeddings = []
        positive_embeddings = []
        for batch in valid_dataloader:
            embedded_source, embedded_target = model(batch, duplicate=False)
            anchor_embeddings.append(embedded_source.cpu().numpy())
            positive_embeddings.append(embedded_target.cpu().numpy())

        # concat
        anchor_embeddings = np.concatenate(anchor_embeddings, axis=0)
        positive_embeddings = np.concatenate(positive_embeddings, axis=0)

        # save to file
        np.save(anchor_root + '/' + name + '.npy', anchor_embeddings)
        np.save(positive_root + '/' + name + '.npy', positive_embeddings)
