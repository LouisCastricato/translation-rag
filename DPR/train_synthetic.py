# This file implements training routines for synthetic datasets

from dataset import *
from model import *
from torch.utils.data import DataLoader
from train_utils import train, embed_valid
from tqdm import tqdm

if __name__ == "__main__":

    inpt_embd_size = 75
    outpt_embd_size = inpt_embd_size
    amnts = np.linspace(100., 1000., 20)[:1]
    noises = [list(np.linspace(1e-5, 3., 20))[10]]

    noise_arr = np.linspace(1e-5, 5., 400)

    for amt, noise in tqdm(zip(amnts, noises)):
        # initialize model and dataset loader
        model = SourceTargetDPR(inpt_embd_size, outpt_embd_size, dropout=0.1).cuda()

        train_dataset = KnownDimensionalityDataset(int(100), val_split=-.5, seed=42)
        validation_datset = KnownDimensionalityDataset(int(100), is_val=True, val_split=.5, seed=42)

        train_dataloader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
        validation_dataloader = DataLoader(validation_datset, batch_size=2048, shuffle=True)

        model = train(model, train_dataloader, validation_loader=validation_dataloader, epochs=100, noise=noise_arr, embed=2)
        #embed_valid(model, validation_dataloader, 'clusters_' + str(noise), anchor_root='time_anchor_embeddings'
