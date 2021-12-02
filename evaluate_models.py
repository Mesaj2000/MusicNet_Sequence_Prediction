from glob import glob
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from util import MODELS_DIR, NUM_COMPOSER_CLASSES
from vector_dataset import Note_Classifier


def test_direct(model, dataloader, device):
    correct = 0
    total = len(dataloader.dataset)

    for data, target in tqdm(dataloader, desc='Direct'):
        data, target = data.to(device).float(), target.to(device).float()

        output = model(data)

        prediction = torch.argmax(output).item()
        ground_truth = torch.argmax(target).item()
        if prediction == ground_truth:
            correct += 1

    print(correct / total)


def test_hierarchical(composer_model, note_models, dataloader, device):
    correct = 0
    total = len(dataloader.dataset)

    for data, target in tqdm(dataloader, desc='Hierarchical'):
        data, target = data.to(device).float(), target.to(device).float()

        composer = composer_model(data)
        composer = torch.argmax(composer).item()
        
        note_model = note_models[composer]
        output = note_model(data)

        prediction = torch.argmax(output).item()
        ground_truth = torch.argmax(target).item()
        if prediction == ground_truth:
            correct += 1

    print(correct / total)



def main():
    device = torch.device('cuda')

    composer_agnostic_note_classifier = torch.load(f'{MODELS_DIR}notes_all.pt').to(device)
    composer_agnostic_note_classifier.eval()

    composer_classifier = torch.load(f'{MODELS_DIR}composer.pt').to(device)
    composer_classifier.eval()

    composer_specific_note_classifiers = []
    for i in range(NUM_COMPOSER_CLASSES):
        model = torch.load(f'{MODELS_DIR}notes_{i}.pt').to(device)
        model.eval()
        composer_specific_note_classifiers.append(model)


    dataset = Note_Classifier(train=False, k=100)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    with torch.no_grad():
        test_direct(composer_agnostic_note_classifier, loader, device)
        test_hierarchical(composer_classifier, composer_specific_note_classifiers, loader, device)


if __name__ == "__main__":
    main()


"""
RESULTS:
Direct:         63.09%
Hierarchical:   44.29%
"""
