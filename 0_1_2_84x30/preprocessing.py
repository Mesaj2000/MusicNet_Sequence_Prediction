import numpy as np
import pandas as pd
import cv2
import os
from intervaltree import Interval, IntervalTree
from pprint import pprint
from glob import glob
from tqdm import tqdm
from util import *
from note_sequence import Note_Sequence

USEFUL_COLUMNS = ('note', 'start_beat', 'end_beat')


# Main preprocessing thing
def load_note_intervals(song: int or str):
    if isinstance(song, int):
        song_path = get_song_path(song)
    elif isinstance(song, str):
        song_path = song

    df = pd.read_csv(song_path, usecols=USEFUL_COLUMNS)


    # DUMMY CODE: GET AVERAGE NOTE LENGTH

    # TODO here: Clean up end_beat to make regular numbers
    # Maybe use note_value to make that happen
    # Ideas:
    # Round to 2 decimal places
    # Set all 16ths to average of 16ths (etc for other values)
    # Find the value closest to 1, scale up up accordingly, scale others by same value?
    # Manually do string matching-based checks: define all notes based on the quarter, 
    #   set quarter based on rounding to nearest half-beat

    # Totally different idea: use start and end TIMES instead of beats
    # Good option for the image approach with CNNs, just have to scale everything
    # Also preserves tempo
    # Problem: how to "extract k beats"?

    # Make the song start at 0
    df["start_beat"] = df["start_beat"] - df["start_beat"].min()

    # Scale values automatically so that the 10th percentile note is 1
    # DANGER: different songs will have different scale values
    # Counter argument: They already do! (some have 16th as 1 beat, others have quarter as 1)
    scale = int(1 / df['end_beat'].quantile(0.1))
    df[['start_beat', 'end_beat']] = df[['start_beat', 'end_beat']] * scale

    # Make end_beat the actual beat where the note ends, not the length of the note
    df['end_beat'] = df['start_beat'] + df['end_beat']

    # Round to integer
    df[['start_beat', 'end_beat']] = df[['start_beat', 'end_beat']].round(0).astype(int)

    # If start == end, it's a "Null Iterval" and it can't go in the tree
    # So, wherever this happens, just increment end by 1
    null_beats = df['start_beat'] == df['end_beat'] # Boolean column
    null_beats = null_beats.astype(int) # 1 if true, 0 if false
    df['end_beat'] = df['end_beat'] + null_beats # Adds 1 where start == end, unchanged for others

    # Adjust note values so that the global min is 0
    df['note'] = df['note'] - LOWESET_NOTE

    # Reorder the columns
    df = df[['start_beat', 'end_beat', 'note']]

    # Select the 

    as_tuple = df.itertuples(index=False, name=None)
    #as_tuple = [(start, start + end, (note)) for note, start, end in as_tuple]
    #as_tuple = [(start, start + end, (note)) for note, start, end in as_tuple]

    tree = IntervalTree.from_tuples(as_tuple)

    return tree


def extract_notes(notes: IntervalTree, k: int = DEFAULT_K, start: int = 0) -> list[Interval]:
    notes_as_list = sorted(notes, key=lambda x: (x.begin, x.end))
    k_notes = notes_as_list[start:start+k]
    return k_notes


def extract_beats(notes: IntervalTree, k: int = DEFAULT_K, start: int = 0) -> list[Interval]:
    k_beats = notes[start:start+k]
    k_beats = sorted(k_beats, key=lambda x: (x.begin, x.end))
    return k_beats


def segment_song(song_number: int, k: int = DEFAULT_K) -> list[Note_Sequence]:
    all_notes = load_note_intervals(song_number)

    num_segments = int(round(all_notes.end() / k))

    sequences = [Note_Sequence(extract_beats(all_notes, k+1, i*k), i*k, k, song_number)
                 for i in range(num_segments)]

    sequences = [s for s in sequences if not s.is_empty()]

    return sequences


def convert_entire_song_to_one_matrix(tree):
    notes = sorted(tree, key=lambda x: (x.begin, x.end, x.data))
    matrix = np.zeros((tree.end(), NOTE_RANGE), dtype=np.float64)

    for note in notes:
        start = note.begin
        end = note.end

        matrix[start:end, note.data] = 1.0
        #matrix[start, note.data] = 1.0

    return matrix


def save_as_image(matrix, song_number):
    im = matrix * 255
    im = np.rot90(im)
    
    path = f'{IMAGE_DIR}{song_number}.bmp'
    guarentee_path_exists(path)

    cv2.imwrite(path, im)


def save_matrix(matrix, song_number):
    matrix_path = f'{NOTE_VECTOR_DIR}{song_number}.npy'
    guarentee_path_exists(matrix_path)

    np.save(matrix_path, matrix)


if __name__ == "__main__":
    guarentee_path_exists(NOTE_VECTOR_DIR)
    all_song_paths = glob(f'{SONGS_DIR}*.csv')
    for song_path in tqdm(all_song_paths):
        song_number = os.path.basename(song_path).split('.')[0]
        tree = load_note_intervals(song_path)
        matrix = convert_entire_song_to_one_matrix(tree)
        save_matrix(matrix, song_number)
        save_as_image(matrix, song_number)
    
    
    
    
    #tree = load_note_intervals(2322, 1)
    #pprint(sorted(tree))


