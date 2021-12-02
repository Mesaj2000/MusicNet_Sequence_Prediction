import os
import shutil
import random
import numpy as np
import pandas as pd
from pprint import pprint
from glob import glob

from util import *


SELECTED_COMPOSERS = [
    "Bach",
    "Beethoven",
    "Brahms",
    "Mozart",
    "Schubert"
]



"""
The major difficulty with the train/test split is that there is very strong class
imbalance between the number of composers. This is arguably not important for the
direct system, but is of critical importance for the hierarchical system and I
want both systems to use the same training and testing data.

Extend of the class imbalance:
    Beethoven: 14,627 samples
    Haydn:     189 samples


New plan: build a composer-agnostic preliminary system, just to get an idea
            what we're working with
"""

def report_num_samples_per_composer(console=True):
    composer_df = pd.read_csv(METADATA_PATH)
    composer_df = composer_df[['id', 'composer']]

    composers = composer_df['composer'].unique()

    composer_to_songs_map = dict()

    for composer in composers:
        composer_to_songs_map[composer] = composer_df[composer_df['composer'] == composer]['id'].tolist()

    num_samples_per_composer = {composer:0 for composer in composers}
    total_samples = 0

    for composer, songs in composer_to_songs_map.items():
        for song in songs:
            num_samples = len(os.listdir(f'{SEQUENCE_DIR}{song}'))
            num_samples_per_composer[composer] += num_samples
            total_samples += num_samples

    if console:
        #pprint(composer_to_songs_map)
        pprint(sorted(list(num_samples_per_composer.items()), key=lambda x: -x[1]))
        print(total_samples)

    return composer_to_songs_map, num_samples_per_composer

"""
Randomly select one-sixth of samples for testing, rest for training
For the current sample set that means: 5745 test + 28728 train = 34473 total
"""
def composer_agnostic_train_test_split():
    # First, load all the samples
    sample_paths = glob(f'{SEQUENCE_DIR}*/*.npy')
    samples = [(sample, sample.replace('saved_sequences', 'sequence_labels')) for sample in sample_paths]
    
    total_samples = len(samples)

    num_testing = total_samples // TESTING_RATIO

    test_set = random.sample(samples, num_testing)
    train_set = [sample for sample in samples if sample not in test_set]

    for data_dir in (TEST_DIR, TRAIN_DIR):
        for sub_dir in ('data/', 'labels/'):
            if guarentee_path_exists(f'{data_dir}{sub_dir}'):
                shutil.rmtree(f'{data_dir}{sub_dir}')
                guarentee_path_exists(f'{data_dir}{sub_dir}')

    for i, (data, label) in enumerate(test_set):
        shutil.copyfile(data, f'{TEST_DIR}data/{i}.npy')
        shutil.copyfile(label, f'{TEST_DIR}labels/{i}.npy')

    for i, (data, label) in enumerate(train_set):
        shutil.copyfile(data, f'{TRAIN_DIR}data/{i}.npy')
        shutil.copyfile(label, f'{TRAIN_DIR}labels/{i}.npy')



# 6 categories: Bach, Beethoven, Brahms, Mozart, Schubert, Other
def composer_classifier_train_test_split():
    try:
        shutil.rmtree(TRAIN_DIR)
        shutil.rmtree(TEST_DIR)
    except FileNotFoundError:
        pass

    guarentee_path_exists(TRAIN_DIR)
    guarentee_path_exists(TEST_DIR)
    
    composer_to_songs_map, num_samples_per_composer = report_num_samples_per_composer(False)
    
    ### TODO: DONT DO THIS HERE, DO IT WHEN LOADING THE DATASET ###
    #samples_per_class = min(num_samples_per_composer.values())
    #testing_samples_per_class = samples_per_class // TESTING_RATIO
    #training_samples_per_class = samples_per_class - testing_samples_per_class

    all_training_data = []
    all_testing_data = []

    for composer, songs_list in composer_to_songs_map.items():
        if composer in SELECTED_COMPOSERS:
            label = SELECTED_COMPOSERS.index(composer)
        else:
            label = len(SELECTED_COMPOSERS)

        train_path = f'{TRAIN_DIR}{label}/{"{}"}.npy'
        test_path = f'{TEST_DIR}{label}/{"{}"}.npy'

        guarentee_path_exists(train_path)
        guarentee_path_exists(test_path)

        samples = []
        for song in songs_list:
            samples.extend(glob(f'{SEQUENCE_DIR}{song}/*.npy'))
        
        total_samples = len(samples)
        num_testing = total_samples // TESTING_RATIO

        test_set = random.sample(samples, num_testing)
        train_set = [sample for sample in samples if sample not in test_set]

        for i, sample in enumerate(test_set):
            shutil.copyfile(sample, test_path.format(i))

        for i, sample in enumerate(train_set):
            shutil.copyfile(sample, train_path.format(i))

        


if __name__ == "__main__":
    report_num_samples_per_composer()