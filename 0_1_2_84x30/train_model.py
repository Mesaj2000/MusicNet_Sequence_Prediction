import numpy as np
import pandas as pd
import traceback

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader

from tqdm import tqdm

from preliminary_model import Basic_CNN, Basic_FC
#from autoencoder import FC_Autoencoder, display_encoder_results
import music_datasets
from util import *
from jaccard import jaccard, jaccard_loss, JaccardDistanceLoss

LEARNING_RATE = 0.01
BATCH_SIZE = 10
NUM_EPOCHS = 50

def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device {device}")

    model = Basic_FC()
    model = model.to(device)

    #criterion = JaccardDistanceLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    print("Loading Datasets")
    train_set = music_datasets.Composer_Classifier(train=True)
    test_set = music_datasets.Composer_Classifier(train=False)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    performance_data = {
        'Train Loss': [],
        'Train Accuracy': [],
        'Test Loss': [],
        'Test Accuracy': []
    }

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_or_test(model, device, criterion, optimizer, train_loader, True)
        test_loss, test_acc = train_or_test(model, device, criterion, optimizer, test_loader, False)
        #display_encoder_results(model, device, test_set)

        performance_data['Train Loss'].append(train_loss)
        performance_data['Train Accuracy'].append(train_acc)
        performance_data['Test Loss'].append(test_loss)
        performance_data['Test Accuracy'].append(test_acc)

        out_str = 'Epoch {:>2}:\n    Train Loss: {:.4f}\n    Train Accu: {:.4f}\n    Test Loss:  {:.4f}\n    Test Accu:  {:.4f}'
        print(out_str.format(epoch, train_loss, train_acc, test_loss, test_acc))
        pd.DataFrame(performance_data).to_csv(f'{TRAINING_OUTPUT_DIR}performance_data.csv')



def train_or_test(model, device, criterion, optimizer, loader, train):
    if train:
        model.train()
        return run_batches(model, device, criterion, optimizer, loader, train)
    else:
        model.eval()
        with torch.no_grad():
            return run_batches(model, device, criterion, optimizer, loader, train)


def run_batches(model, device, criterion, optimizer, loader, train):
    losses = []
    accuracies = []
    correct = 0

    for data, target in tqdm(loader, desc='Training' if train else 'Testing'):
        data, target = data.to(device).float(), target.to(device)
        
        #data = (data > 0).float() # THIS IS A HACK!!! PLESASE DON"T LEAVE THIS IN THE FINAL VERSION
        #target = (target > 0).float()

        if train:
            optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)
        #print(loss)

        if train:
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        prediction = output.argmax(dim=1, keepdim=False)
        target = target.argmax(dim=1, keepdim=False)
        #prediction = torch.zeros(output.shape).to(device)
        #target = torch.round(target)
        #accuracy = jaccard(prediction, target)
        #accuracy = (prediction == target).float().mean().item()
        #accuracies.append(0)
        #print(prediction.shape)
        #print(target.shape)
        correct += (prediction == target).sum().item()

    avg_loss = np.mean(losses)
    #avg_accu = np.mean(accuracies)
    avg_accu = correct / len(loader.dataset)
    return avg_loss, avg_accu



def make_plots():
    performance_data = pd.read_csv(f'{TRAINING_OUTPUT_DIR}performance_data.csv')
    x = list(range(len(performance_data)))
    
    fig, (loss_ax, accuracy_ax) = plt.subplots(2, sharex=True)
    
    ### Loss Plots ###
    train_loss = performance_data['Train Loss']
    test_loss = performance_data['Test Loss']

    loss_ax.plot(x, train_loss, c='red', label='Training')
    loss_ax.plot(x, test_loss, c='blue', label='Testing')
    
    loss_ax.set_title('Loss')
    loss_ax.legend()
    
    ### Accuracy Plots ###
    train_accuracy = performance_data['Train Accuracy']
    test_accuracy = performance_data['Test Accuracy']
    
    accuracy_ax.plot(x, train_accuracy, c='red', label='Training')
    accuracy_ax.plot(x, test_accuracy, c='blue', label='Testing')
    
    accuracy_ax.set_title('Accuracy')
    accuracy_ax.set_xlabel('Epochs')
    accuracy_ax.legend()
    
    
    ### Save the plots to disk ###
    fig.savefig(f'{TRAINING_OUTPUT_DIR}performance_data_plot.jpg', dpi=1000, bbox_inches='tight')


if __name__ == "__main__":
    guarentee_path_exists(TRAINING_OUTPUT_DIR)
    
    try:
        main()
    except Exception as e:
        traceback_string = ''.join(traceback.format_exception(None, e, e.__traceback__))
        print(traceback_string)

    #make_plots()