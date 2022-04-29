import torch

# implements a single layer DPR model
class SingeLayerDPR(torch.nn.Module):
    def __init__(self, input_size, output_size, dropout=0.0):
        super(SingeLayerDPR, self).__init__()

        # Linear, ReLu, dropout
        self.project = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.Dropout(dropout))
        
        self.input_size = input_size
        self.output_size = output_size
    
    def forward(self, x):
        return self.project(x)
    
    