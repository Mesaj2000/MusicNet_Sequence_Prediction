import torch
from torch import nn
from torch.nn import functional as F

from util import *


INPUT_HEIGHT = 84
INPUT_WIDTH = 30

OUTPUT_SIZE = 6

NUM_KERNELS = 30
KERNEL_SIZE = 7
STRIDE = 1
PADDING = (KERNEL_SIZE - 1) // 2

NUM_CONV_LAYERS = 4
NUM_FC_LAYERS = 2
POOL_SIZE = 2
POOL_PADDING = 1

POOLED_HEIGHT = INPUT_HEIGHT
for _ in range(NUM_CONV_LAYERS):
    POOLED_HEIGHT = (POOLED_HEIGHT + 2 * POOL_PADDING) // POOL_SIZE

POOLED_WIDTH = INPUT_WIDTH
for _ in range(NUM_CONV_LAYERS):
    POOLED_WIDTH = (POOLED_WIDTH + 2 * POOL_PADDING) // POOL_SIZE

FLATTENED_SIZE = NUM_KERNELS * POOLED_HEIGHT * POOLED_WIDTH

ACTIVATION_FUNCTION = torch.sigmoid


class Basic_CNN(nn.Module):
    def __init__(self):
        super(Basic_CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, NUM_KERNELS, KERNEL_SIZE, STRIDE, PADDING)
        self.conv2 = nn.Conv2d(NUM_KERNELS, NUM_KERNELS, KERNEL_SIZE, STRIDE, PADDING)
        self.conv3 = nn.Conv2d(NUM_KERNELS, NUM_KERNELS, KERNEL_SIZE, STRIDE, PADDING)
        self.conv4 = nn.Conv2d(NUM_KERNELS, NUM_KERNELS, KERNEL_SIZE, STRIDE, PADDING)

        self.fc1 = nn.Linear(FLATTENED_SIZE, FLATTENED_SIZE)
        self.fc2 = nn.Linear(FLATTENED_SIZE, OUTPUT_SIZE)

        self.conv_layers = [self.conv1, self.conv2, self.conv3, self.conv4]
        self.fc_layers = [self.fc1, self.fc2]

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)

        for conv in self.conv_layers:
            x = conv(x)
            x = F.max_pool2d(x, POOL_SIZE, padding=POOL_PADDING)
            x = ACTIVATION_FUNCTION(x)

        x = torch.flatten(x, start_dim=1)

        for fc in self.fc_layers:
            x = fc(x)
            x = ACTIVATION_FUNCTION(x)

        return x



class Basic_FC(nn.Module):
    def __init__(self, num_layers=NUM_FC_LAYERS):
        super(Basic_FC, self).__init__()

        self.num_layers = num_layers

        input_size = INPUT_HEIGHT * INPUT_WIDTH
        hidden_size = input_size // 2

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc_layers = [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]

        self.fc_final = nn.Linear(hidden_size, OUTPUT_SIZE)


    def forward(self, x):
        x = torch.flatten(x, start_dim=1)

        for fc in self.fc_layers[:self.num_layers]:
            x = fc(x)
            x = ACTIVATION_FUNCTION(x)

        x = self.fc_final(x)
        x = F.softmax(x, dim=1)

        return x