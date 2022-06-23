# this file implements bog standard contrastive learning for training a DPR model

from dataset import *
from model import *
from torch.utils.data import DataLoader
from train_utils import train
from tqdm import tqdm

if __name__ == "__main__":

    inpt_embd_size = 300
    outpt_embd_size = 300

    # initialize model and dataset loader
    model = SourceTargetDPR(inpt_embd_size, outpt_embd_size, dropout=0.0)
    dataset = ProcessedPairedTextDataset('rag-processed-datasets/index-vocab/en-es.json')
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    train(model, dataloader)

    # save the model
    torch.save(model.state_dict(), 'DPR_encoder.pt')