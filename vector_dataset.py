import os
import random
from glob import glob
from pprint import pprint
from numpy.core.numeric import indices
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from intervaltree import Interval, IntervalTree
from matplotlib import pyplot as plt
from train_test_split_lstm import SELECTED_COMPOSERS
from util import *

"""
TODO MAYBE:
Only need one dataset class
Returns both the prediction AND the composer
    (maybe omit the composer based on an init argument)
    (maybe also assure composer classes are balanced based on argument)
"""

class Composer_Agnositc(Dataset):
    def __init__(self, train=True, k=DEFAULT_K):
        data_dir = TRAIN_DIR if train else TEST_DIR
        vector_paths = glob(f'{data_dir}*/*.npy')

        interval_tree_of_vector_lists = IntervalTree()
        end_pos_of_most_recent_vector_list = 0
        for path in vector_paths:
            vector_list = np.load(path)
            start_pos_of_this_vector_list = end_pos_of_most_recent_vector_list
            end_pos_of_this_vector_list = start_pos_of_this_vector_list + len(vector_list) - k
            vectors_as_interval = Interval(start_pos_of_this_vector_list,
                                           end_pos_of_this_vector_list,
                                           vector_list)
            interval_tree_of_vector_lists.add(vectors_as_interval)
            end_pos_of_most_recent_vector_list = end_pos_of_this_vector_list

        self.vector_tree = interval_tree_of_vector_lists
        self.final_position = end_pos_of_most_recent_vector_list
        self.k = k

    def __len__(self):
        return self.final_position

    def __getitem__(self, index):
        vector_list_interval, = sorted(self.vector_tree[index])
        start = vector_list_interval.begin
        vector_list = vector_list_interval.data
        index_into_list = index - start
        vector_sequence = vector_list[index_into_list:index_into_list + self.k]
        subsequent_vector = vector_list[index_into_list + self.k]
        label = np.zeros(NOTE_RANGE)
        label[round(subsequent_vector[4] * (HIGHEST_NOTE - LOWESET_NOTE))] = 1
        return vector_sequence, label



class Note_Classifier(Dataset):
    def __init__(self, train=True, k=DEFAULT_K, composer='*', verbose=True):

        assert composer in list('012345*') or composer in range(6), "Invalid composer argument to Note_Classifier.__init__"

        data_dir = TRAIN_DIR if train else TEST_DIR
        song_matrix_paths = glob(f'{data_dir}{composer}/*.npy')
        song_matrix_paths.sort(key=lambda x: int(os.path.basename(x).replace('.npy', '')))

        interval_tree_of_songs = IntervalTree()
        end_pos_of_most_recent_song_interval = 0
        for path in song_matrix_paths:
            song_matrix = np.load(path)
            start_pos_of_this_song_interval = end_pos_of_most_recent_song_interval
            end_pos_of_this_song_interval = start_pos_of_this_song_interval + len(song_matrix) - k
            song_interval = Interval(start_pos_of_this_song_interval,
                                    end_pos_of_this_song_interval,
                                    song_matrix)
            interval_tree_of_songs.add(song_interval)
            end_pos_of_most_recent_song_interval = end_pos_of_this_song_interval

        self.song_interval_tree = interval_tree_of_songs
        self.final_position = end_pos_of_most_recent_song_interval
        self.k = k

        index_map = {note:[] for note in range(NOTE_PREDICTION_RANGE)}

        # Iterate over every sample (this will be slow)
        for index in tqdm(range(self.final_position), 
                                desc=f'Loading {"Training" if train else "Testing"} Dataset',
                                disable=not verbose):
            # Categorize samples based on label
            try:
                slice, label = self.get_item_from_entire_dataset(index)
                target_note = np.nonzero(label)[0][0]
                index_map[target_note].append(index)
            except (IndexError, AssertionError):
                pass


        index_map_counter = [len(indexes) for note, indexes in index_map.items()]
        #print(min(index_map_counter))
        #pprint(index_map_counter)
        #pprint(np.argwhere((np.array(index_map_counter) / self.final_position) > 0.005))
        #pprint(np.all((np.array(index_map_counter) / self.final_position) > 0.005))
        #plt.scatter(range(len(index_map_counter)), index_map_counter)
        #plt.show()

        # Balance the labels
        self.selected_indexes = []
        samples_per_class = min(index_map_counter)
        for note, indexes in index_map.items():
            self.selected_indexes.extend(random.sample(indexes, samples_per_class))
        
        random.shuffle(self.selected_indexes)


    def get_item_from_entire_dataset(self, index):
        song_interval, = list(self.song_interval_tree[index])
        start = song_interval.begin
        song_matrix = song_interval.data
        index_into_matrix = index - start
        slice = song_matrix[index_into_matrix:index_into_matrix + self.k]
        subsequent_beat = song_matrix[index_into_matrix + self.k]
        
        # Change second [0] to [-1] to target the highest note
        target_note = np.nonzero(subsequent_beat)[0][0]
        assert LOWEST_PREDICTED_NOTE <= target_note <= HIGHEST_PREDICTED_NOTE
        
        label = np.zeros(NOTE_PREDICTION_RANGE)
        label[target_note - LOWEST_PREDICTED_NOTE] = 1
        return slice, label

    
    def __len__(self):
        return len(self.selected_indexes)

    def __getitem__(self, index):
        true_index = self.selected_indexes[index]
        return self.get_item_from_entire_dataset(true_index)



class Composer_Classifier(Dataset):
    def __init__(self, train=True, k=DEFAULT_K, composer='dummyarg'):
        data_dir = TRAIN_DIR if train else TEST_DIR
        composer_trees = []
        for composer in os.listdir(data_dir):
            song_paths = glob(f'{data_dir}{composer}/*.npy')
            tree = IntervalTree()
            prev_end_pos = 0
            for path in song_paths:
                song_matrix = np.load(path)
                start_pos = prev_end_pos
                end_pos = start_pos + len(song_matrix) - k
                song_interval = Interval(start_pos,
                                         end_pos,
                                         song_matrix)
                tree.add(song_interval)
                prev_end_pos = end_pos
            composer_trees.append(tree)

        min_samples = min(composer_trees, key=lambda x: x.end()).end()

        for tree in composer_trees:
            tree.chop(min_samples, tree.end())
        
        self.samples_per_composer = min_samples
        self.composer_trees = composer_trees
        self.num_composers = len(composer_trees)
        self.total_samples = self.samples_per_composer * self.num_composers
        self.k = k

    def __len__(self):
        return self.total_samples

    def __getitem__(self, index):
        composer = index // self.samples_per_composer
        index_relative_to_composer = index % self.samples_per_composer
        tree = self.composer_trees[composer]
        song_interval, = list(tree[index_relative_to_composer])
        start = song_interval.begin
        song_matrix = song_interval.data
        index_relative_to_song = index_relative_to_composer - start
        note_sequence = song_matrix[index_relative_to_song:index_relative_to_song + self.k]
        label = np.zeros(self.num_composers)
        label[composer] = 1
        return note_sequence, label



if __name__ == "__main__":
    """
    dataset = Note_Classifier()
    previous_index = 0
    previous_target_note = 0
    correct = 0
    total = 0
    for index in tqdm(range(dataset.final_position)):
        try:
            slice, label = dataset.get_item_from_entire_dataset(index)
            target_note = np.nonzero(label)[0][0]
            
            if index == previous_index + 1:
                total += 1
                if previous_target_note == target_note:
                    correct += 1

            previous_target_note = target_note
                
        except (IndexError, AssertionError):
            pass
            
        finally:
            previous_index = index


    print(correct / total) # 72.68%
    exit()


    #for c in '012345':
     #   dataset = Note_Classifier(train=False, composer=c)
    #exit()

    dataset = Note_Classifier(train=False, k=100)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    for data, label in dataset:
        print(data, label)

        if i > 2:
            break

    """
    pass