import torch
from torch import nn
from torch.nn import functional as F

from util import *


class Basic_LSTM(nn.Module):
    def __init__(self, input_size, sequence_length, batch_size, output_size):
        super(Basic_LSTM, self).__init__()

        self.input_size = input_size
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.flattened_size = sequence_length * input_size
        self.output_size = output_size

        self.lstm1 = nn.LSTM(input_size=input_size, 
                             hidden_size=input_size, 
                             num_layers=1,
                             batch_first=True,
                             dropout=0)
        
        #self.fc1 = nn.Linear(self.flattened_size, self.flattened_size // 2)
        #self.fc2 = nn.Linear(self.flattened_size // 2, output_size)

        self.fc1 = nn.Linear(self.flattened_size, output_size)


    def forward(self, x):
        x, hidden = self.lstm1(x)
        
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        #x = F.relu(x)
        
        #x = self.fc2(x)
        #x = torch.softmax(x, dim=1)
        x = torch.squeeze(x)
        return x


if __name__ == "__main__":
    model = Basic_LSTM(input_size=NOTE_RANGE, 
                       sequence_length=100, 
                       batch_size=128, 
                       output_size=NOTE_PREDICTION_RANGE)
    print(model)
