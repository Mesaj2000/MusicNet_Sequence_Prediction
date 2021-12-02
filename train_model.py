import numpy as np
import pandas as pd
import traceback
from datetime import datetime

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader

from tqdm import tqdm

from lstm_models import Basic_LSTM
#from autoencoder import FC_Autoencoder, display_encoder_results
from vector_dataset import Note_Classifier, Composer_Classifier
from util import *

LEARNING_RATE = 0.05
REGULARIZATION_WEIGHT = 0
MOMENTUM = 0.9

BATCH_SIZE = 128
NUM_EPOCHS = 20

TIMESTAMP = datetime.now().isoformat().replace('T', '_').replace(':', '-').split('.')[0]
OUTPUT_DIR = f'{TRAINING_OUTPUT_DIR}{TIMESTAMP}/'

SEQUENCE_LEN = 100

def main(predict_notes=True, composer='*'):
    if predict_notes:
        assert composer in list('012345*') or composer in range(NUM_COMPOSER_CLASSES), "Invalid composer argument to main"

    output_path = f'{OUTPUT_DIR}{filename_generator(predict_notes, composer)}'

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device {device}")

    output_size = NOTE_PREDICTION_RANGE if predict_notes else NUM_COMPOSER_CLASSES
    
    model = Basic_LSTM(input_size=NOTE_RANGE, 
                       sequence_length=SEQUENCE_LEN, 
                       batch_size=BATCH_SIZE, 
                       output_size=output_size)
    
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=REGULARIZATION_WEIGHT, momentum=MOMENTUM)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=3, threshold=0.01)

    dataset_class = Note_Classifier if predict_notes else Composer_Classifier

    #train_set = dataset_class(train=True, k=SEQUENCE_LEN, composer=composer)
    #test_set = dataset_class(train=False, k=SEQUENCE_LEN, composer=composer)

    #train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    #test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    performance_data = {
        'Train Loss': [],
        'Train Accuracy': [],
        'Test Loss': [],
        'Test Accuracy': []
    }

    best_test_acc = 0

    for epoch in range(NUM_EPOCHS):
        train_set = dataset_class(train=True, k=SEQUENCE_LEN, composer=composer)
        test_set = dataset_class(train=False, k=SEQUENCE_LEN, composer=composer)
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        train_loss, train_acc = train_or_test(model, device, criterion, optimizer, train_loader, True)
        test_loss, test_acc = train_or_test(model, device, criterion, optimizer, test_loader, False)
        current_lr = optimizer.param_groups[0]['lr']

        performance_data['Train Loss'].append(train_loss)
        performance_data['Train Accuracy'].append(train_acc)
        performance_data['Test Loss'].append(test_loss)
        performance_data['Test Accuracy'].append(test_acc)

        out_str = 'Epoch {:>2}:\n    Learning Rate: {:.4}\n    Train Loss: {:.4f}\n    Train Accu: {:.4f}\n    Test Loss:  {:.4f}\n    Test Accu:  {:.4f}'
        print(out_str.format(epoch, current_lr, train_loss, train_acc, test_loss, test_acc))
        pd.DataFrame(performance_data).to_csv(f'{output_path}.csv')
        make_plots(path=output_path, dpi=1000)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model, f'{output_path}.pt')

        scheduler.step(test_loss)


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
    #accuracies = []
    correct = 0

    for data, target in tqdm(loader, desc='Training' if train else 'Testing'):
        data, target = data.to(device).float(), target.to(device).float()

        if train:
            optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)

        if train:
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        prediction = torch.argmax(output, dim=1)
        ground_truth = torch.argmax(target, dim=1)
        good_predictions = (prediction == ground_truth)
        correct += good_predictions.sum().item()

    avg_loss = np.mean(losses)
    #avg_accu = np.mean(accuracies)
    avg_accu = correct / (len(loader.dataset))
    return avg_loss, avg_accu



def make_plots(path, dpi=1000):
    performance_data = pd.read_csv(f'{path}.csv')
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
    fig.savefig(f'{path}.jpg', dpi=dpi, bbox_inches='tight')
    plt.close()


def filename_generator(predict_notes, composer):
    if composer == '*':
        composer = 'all'
    return f"notes_{composer}" if predict_notes else "composer"


if __name__ == "__main__":
    guarentee_path_exists(OUTPUT_DIR)

    try:
        main(predict_notes=False)
    except Exception as e:
        traceback_string = ''.join(traceback.format_exception(None, e, e.__traceback__))
        print(traceback_string)


    for composer in '012345*':
        try:
            main(predict_notes=True, composer=composer)
        except Exception as e:
            traceback_string = ''.join(traceback.format_exception(None, e, e.__traceback__))
            print(traceback_string)


