import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from music_datasets import Music_Dataset
from util import *

INPUT_HEIGHT = 84
INPUT_WIDTH = 30

INPUT_SIZE = INPUT_HEIGHT * INPUT_WIDTH # 84x30 = 2520
#OUTPUT_SIZE = INPUT_HEIGHT

INTERMEDIATE_SIZE = 1600
ENCODED_SIZE = 1000

IMAGE_STR = f"{TRAINING_OUTPUT_DIR}autoencoder_{'{}'}.png"


class FC_Autoencoder(nn.Module):
    def __init__(self):
        super(FC_Autoencoder, self).__init__()

        self.encode1 = nn.Linear(INPUT_SIZE, INTERMEDIATE_SIZE)
        self.encode2 = nn.Linear(INTERMEDIATE_SIZE, ENCODED_SIZE)

        self.decode1 = nn.Linear(ENCODED_SIZE, INTERMEDIATE_SIZE)
        self.decode2 = nn.Linear(INTERMEDIATE_SIZE, INPUT_SIZE)


    def encode(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.encode1(x)
        x = self.encode2(x)
        return x

    def decode(self, x):
        x = self.decode1(x)
        x = self.decode2(x)
        x = torch.reshape(x, (-1, INPUT_HEIGHT, INPUT_WIDTH))
        return x

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return encoded, decoded



# Find 10 of each digit and encode/decode for the report
def display_encoder_results(model, device, test_set):
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2)

    for i, (image, labels) in enumerate(test_loader):
        image = image.to(device).float()
        
        
        # Must reshape exactly as we do in the training
        #image = torch.reshape(image, (-1, INPUT_HEIGHT, INPUT_WIDTH))
        
        
        encoded, decoded = model(image)
        
        
        # Appends the original and the reconstructed into one image
        concatenated = torch.cat((image, decoded), 2)
        save_image(concatenated, IMAGE_STR.format(i))
        #saved_images.append(concatenated)
        
        
        # Stop when we have all 10 desired images
        if i >= 5:
            break


if __name__ == "__main__":
    print(IMAGE_STR.format(0))