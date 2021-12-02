import os
import shutil
import random
import numpy as np
import pandas as pd
from pprint import pprint
from glob import glob

from util import *

### For now, don't worry about this ###
SELECTED_COMPOSERS = [
    "Bach",
    "Beethoven",
    "Brahms",
    "Mozart",
    "Schubert",
    "Other"
]

vector_path = f'{NOTE_VECTOR_DIR}{"{}"}.npy'
train_path = f'{TRAIN_DIR}{"{}"}/{"{}"}.npy'
test_path = f'{TEST_DIR}{"{}"}/{"{}"}.npy'


def copy_note_vectors_to_train_and_test_folders():
    composer_df = pd.read_csv(METADATA_PATH)
    composer_df = composer_df[['id', 'composer']]

    composers = composer_df['composer'].unique()

    composer_to_songs_map = dict()


    for composer in composers:
        composer_to_songs_map[composer] = composer_df[composer_df['composer'] == composer]['id'].tolist()

    # Create all the directories where we will save the train/test data
    guarentee_path_exists(TRAIN_DIR)
    guarentee_path_exists(TEST_DIR)
    for i, composer in enumerate(SELECTED_COMPOSERS):
        guarentee_path_exists(train_path.format(i, 0))
        guarentee_path_exists(test_path.format(i, 0))


    # First, copy all the songs to the training directory
    for composer, songs in composer_to_songs_map.items():

        try:
            composer_idx = SELECTED_COMPOSERS.index(composer)
        except ValueError:
            composer_idx = len(SELECTED_COMPOSERS) - 1
        
        for song in songs:
            shutil.copyfile(vector_path.format(song), 
                            train_path.format(composer_idx, song))


    # Second, move some of the songs into the testing directory
    # This is in two steps because some composers are first merged into "Other"
    for i, composer in enumerate(SELECTED_COMPOSERS):
        all_vector_paths = glob(train_path.format(i, '*'))
        num_test_vectors = len(all_vector_paths) // TESTING_RATIO
        test_vector_paths = random.sample(all_vector_paths,  num_test_vectors)
        for test_vector_path in test_vector_paths:
            basename = os.path.basename(test_vector_path)
            shutil.move(test_vector_path, f'{TEST_DIR}{i}/{basename}')


def output_num_songs_per_composer():
    composer_df = pd.read_csv(METADATA_PATH)
    composer_df = composer_df[['id', 'composer']]

    composers = composer_df['composer'].unique()

    composer_to_songs_map = dict()


    for composer in composers:
        composer_to_songs_map[composer] = composer_df[composer_df['composer'] == composer]['id'].tolist()
    

    for composer, songs in sorted(list(composer_to_songs_map.items()), key=lambda x: -len(x[1])):
        print(f"{composer:<11}{len(songs):>6}")

if __name__ == "__main__":
    #copy_note_vectors_to_train_and_test_folders()
    output_num_songs_per_composer()
    pass
