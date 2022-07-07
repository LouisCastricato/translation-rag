# this file implements bog standard contrastive learning for training a DPR model

from dataset import *
from model import *
from torch.utils.data import DataLoader
from train_utils import train, embed_valid
from tqdm import tqdm

if __name__ == "__main__":

    noises = np.linspace(1e-5, 1., 200)
    train_dataset = ProcessedPairedTextDataset('DPR-processed-data/en-es.train.json')
    valid_dataset = ProcessedPairedTextDataset('DPR-processed-data/en-es.valid.json')

    inpt_embd_size = 300
    outpt_embd_size = 300

    # initialize model and dataset loader
    model = SourceTargetDPR(inpt_embd_size, outpt_embd_size, dropout=0.1).cuda()
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    valid_dataloader = DataLoader(valid_dataset, batch_size=256, shuffle=True)

    train(model, train_dataloader, valid_dataloader, epochs=20, noise=noises, embed=1)

    # save the model
    torch.save(model.state_dict(), 'DPR_encoder.pt')
