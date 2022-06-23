# This file implements training routines for synthetic datasets

from dataset import *
from model import *
from torch.utils.data import DataLoader
from train_utils import train
from tqdm import tqdm

if __name__ == "__main__":

    inpt_embd_size = 75
    outpt_embd_size = inpt_embd_size*2
    amnts = np.linspace(100., 1000., 20)

    for amt in tqdm(amnts):
        # initialize model and dataset loader
        model = SourceTargetDPR(inpt_embd_size, outpt_embd_size, dropout=0.1).cuda()

        train_dataset = KnownDimensionalityDataset(int(amt), val_split=-.5, seed=42)
        validation_datset = KnownDimensionalityDataset(int(amt), is_val=True, val_split=.5, seed=42)

        train_dataloader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
        validation_dataloader = DataLoader(validation_datset, batch_size=2048, shuffle=True)

        train(model, train_dataloader, epochs=20)

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


