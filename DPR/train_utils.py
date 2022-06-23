import torch
from tqdm import tqdm

def train(model, dataset_loader, epochs=10):
    optim = torch.optim.AdamW(model.parameters(), lr=5e-2, weight_decay=1e-5)
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
                logits = torch.bmm(embedded_source.unsqueeze(1), embedded_target.transpose(1, 2))
                # softmax
                logits = torch.nn.functional.log_softmax(logits, dim=-1).squeeze()
                # if we used a predefined batch
                if 'target_batch' in contrastive_batch.keys():
                    # sum along the 0th column
                    return -logits[:, 0].mean()
                else:
                    return -logits.diag().mean()

            loss_value = loss(embedded_source, embedded_target)
            loss_value.backward()
            optim.step()
            optim.zero_grad()
        pbar.set_description(f"loss: {loss_value}")