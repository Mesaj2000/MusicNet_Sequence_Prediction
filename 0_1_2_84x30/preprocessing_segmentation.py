import numpy as np
import pandas as pd
import cv2
from intervaltree import Interval, IntervalTree
from pprint import pprint
import os
from matplotlib import pyplot as plt
from shutil import rmtree
from util import *
from note_sequence import Note_Sequence
import preprocessing as pp


def segment_and_save_all(save_image: bool = False) -> None:
    songs = os.listdir(SONGS_DIR)
    songs = [s.split('.', 1)[0] for s in songs]
    songs = [int(s) for s in songs]

    for s in songs:
        segment_and_save(s, save_image=save_image)


def segment_and_save(song_number: int = 2310, 
                     k: int = DEFAULT_K, 
                     save_image: bool = False) -> list[Note_Sequence]:
    
    remove_directories([IMAGE_DIR, SEQUENCE_DIR, LABEL_DIR], song_number)      

    sequences = pp.segment_song(song_number, k)
    for seq in sequences:
        seq.save()
        if save_image:
            seq.save_as_image()
        
    return sequences


def load_matrices(song_number: int = 2310) -> None:
    matrices = os.listdir(f'{SEQUENCE_DIR}{song_number}/')
    matrices.sort(key=lambda x: int(x.split('-', 1)[0]))
    matrices = [f'{SEQUENCE_DIR}{song_number}/{x}' for x in matrices]
    matrices = [np.load(x) for x in matrices]
    return matrices


def remove_directories(dirs: list[str], song_number: int) -> None:
    for d in dirs:
        try:
            rmtree(f'{d}{song_number}/')
        except FileNotFoundError:
            pass


def plot_shapes(matrices: np.ndarray) -> None:
    x = [m.shape[0] for m in matrices]
    y = [m.shape[1] for m in matrices]
    plt.scatter(x, y)
    plt.show()


# I used this to find out that the highest note across all songs is 104
# and the lowest is 21
def find_note_range_extremes() -> None:
    songs = os.listdir(SONGS_DIR)
    songs = [s.split('.', 1)[0] for s in songs]
    songs = [int(s) for s in songs]

    trees = [pp.load_note_intervals(pp.get_song_path(s)) for s in songs]
    list_of_list_of_notes = [sorted(tree, key=lambda x: (x.begin, x.end)) 
        for tree in trees]
    all_notes = [note for l in list_of_list_of_notes for note in l]

    max_note = max(all_notes, key=lambda x: x.data)
    min_note = min(all_notes, key=lambda x: x.data)

    print(max_note)
    print(min_note)


if __name__ == "__main__":
    #segment_and_save_all(True)
    song_number = 2322
    #seqs = segment_and_save(song_number, 30, True)
    matrices = load_matrices(song_number)
    #plot_shapes(matrices)

