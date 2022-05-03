import torch
from torch import nn

# implements a single layer DPR model
class SingeLayerDPR(torch.nn.Module):
    def __init__(self, input_size, output_size, dropout=0.1):
        """
        :param input_size: the size of the input embedding
        :param output_size: the size of the output embedding
        :param dropout: the dropout rate
        """
        super(SingeLayerDPR, self).__init__()

        # Linear, ReLu, dropout
        self.project = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.Dropout(dropout))
        
        self.input_size = input_size
        self.output_size = output_size
    
    def forward(self, x):
        """
        :param x: the input embedding. Of size bs x input_size
        :return: the output embedding. Of size bs x output_size
        """
        return self.project(x)
    
    
class SourceTargetDPR(torch.nn.Module):
    def __init__(self, input_size, output_size, dropout=0.1):
        """
        input_size: the size of the input embedding
        output_size: the size of the output embedding
        dropout: the dropout rate
        """
        super(SourceTargetDPR, self).__init__()
        self.source_encoder = SingeLayerDPR(input_size, output_size, dropout)
        self.target_encoder = SingeLayerDPR(input_size, output_size, dropout)
    
    def forward(self, x):
        """
        x: a tuple of batched source and target language embeddings
        :return: a tuple of batched source and target language embeddings, post projection
        """
        return self.source_encoder(x[0]), self.target_encoder(x[1])