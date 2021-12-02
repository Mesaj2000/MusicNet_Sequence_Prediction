import math
from glob import glob
import numpy as np
import pandas as pd
from intervaltree import Interval, IntervalTree
from tqdm import tqdm
from pprint import pprint
from util import *

USEFUL_COLUMNS = ('note', 'start_beat', 'end_beat')


"""
A "Note Vector" is a vector summarizing core information (features) of
a note relative to the notes around it. Elements in the vector include:
    time since previous start: time between the start of this note and the *start*
        of the most recent note that *started* before this note.
    difference from previous start: the change in note value between the two notes.
        If there is a tie for the previous start note, use the smallest difference.
    
    time since previous end: time between the start of this note and the *end*
        of the most recent note that *end* before this note started.
    difference from previous end: the change in note value between the two notes.
        If there is a tie for the previous start note, use the smallest difference.
    
    note: the value of the note
    duration: the length of this note

Every note in a song is converted to a note vector, which are combined into one
large numpy array for the song. Each note vector is one row of the array.
"""
def form_note_vectors(song):
    if isinstance(song, int):
        song_path = get_song_path(song)
    elif isinstance(song, str):
        song_path = song

    ### First, load all the notes into an interval tree ###
    df = pd.read_csv(song_path, usecols=USEFUL_COLUMNS)

    # "end_beat" in the data is a bad name for it; it's actually the length
    # make a new column for the length, update end_beat to be correct
    df['length'] = df['end_beat']
    df['end_beat'] = df['start_beat'] + df['length']

    # Reorder columns to work well with the intervaltree library
    df = df[['start_beat', 'end_beat', 'note', 'length']]

    # Normalize the data
    df['note'] = (df['note'] - LOWESET_NOTE) / (HIGHEST_NOTE - LOWESET_NOTE)
    df['length'] = (df['length'] - df['length'].min()) / \
                   (df['length'].max() - df['length'].min())
    


    # Convert to interval tree
    as_tuple = df.itertuples(index=False, name=None)
    as_tuple = [(start, end, {'note': note, 'length': length}) 
                for start, end, note, length in as_tuple]
    tree = IntervalTree.from_tuples(as_tuple)

    ### Create the note vectors ###
    prune = tree # this can be set to a slice for debugging purposes
    grouped_by_start = group_intervals_by_start(prune)
    grouped_by_end = group_intervals_by_end(prune)
    vectors = np.ndarray((len(prune), 6), dtype=float)
    vector_pos = 0
    for i, group in enumerate(grouped_by_start):
        for interval in group:
            # Get information about THIS note
            start = interval.begin
            end = interval.end
            note = interval.data['note']
            length = interval.data['length']

            # Get information about the note that started most 
            # recently before this one started
            if i > 0:
                previous_group = grouped_by_start[i-1]
                previous_start_time = previous_group[0].begin
                time_since_previous_start = start - previous_start_time
                difference_from_previous_start = \
                    note - min(previous_group, 
                               key=lambda x: np.abs(note - x.data['note'])
                               ).data['note']
            else:
                time_since_previous_start = 0
                difference_from_previous_start = 0


            # Get information about the note that ended most 
            # recently before this one started
            if grouped_by_end[0][0].end < start:
                end_group_index = search_end_grouped_intervals(grouped_by_end, start)
                previous_end_group = grouped_by_end[end_group_index]
                previous_end_time = previous_end_group[0].end
                time_since_previous_end = start - previous_end_time
                difference_from_previous_end = \
                    note - min(previous_end_group, 
                               key=lambda x: np.abs(note - x.data['note'])
                               ).data['note']
            else:
                time_since_previous_end = 0
                difference_from_previous_end = 0

            
            # Create the note vector as a row in a numpy array
            vectors[vector_pos] = np.array([time_since_previous_start,
                                            difference_from_previous_start,
                                            time_since_previous_end,
                                            difference_from_previous_end,
                                            note,
                                            length], dtype=float)
            vector_pos += 1

    # Normalize time_since_previous_X
    if np.max(vectors[:,0]) > 0:
        vectors[:,0] = (vectors[:,0] - np.max(vectors[:,0])) / np.max(vectors[:,0])
    if np.max(vectors[:,2]) > 0:
        vectors[:,2] = (vectors[:,2] - np.max(vectors[:,2])) / np.max(vectors[:,2])
    return vectors



"""
Input: an interval tree (or a list of intervals)
Output: a list of lists of intervals, where every interval in a sublist
        has the same start beat, and the sublists themselves are in sorted order
"""
def group_intervals_by_start(intervals):
    grouped_intervals = [[]]
    intervals = sorted(intervals, 
                       key=lambda x: (x.begin, x.end, x.data['note'], x.data['length']))

    pos = intervals[0].begin
    while intervals:
        interval = intervals.pop(0)
        if interval.begin == pos:
            grouped_intervals[-1].append(interval)
        else:
            pos = interval.begin
            grouped_intervals.append([interval])
            
    return grouped_intervals

"""
Ditto above, but the groups are based on the end beat
"""
def group_intervals_by_end(intervals):
    grouped_intervals = [[]]
    intervals = sorted(intervals, 
                       key=lambda x: (x.end, x.begin, x.data['note'], x.data['length']))

    pos = intervals[0].end
    while intervals:
        interval = intervals.pop(0)
        if interval.end == pos:
            grouped_intervals[-1].append(interval)
        else:
            pos = interval.end
            grouped_intervals.append([interval])
            
    return grouped_intervals


"""
Input: a list of interval groups, grouped by their end beat
Output: the index into the list of the group that ends at the given beat
(It's just a binary search)
"""
def search_end_grouped_intervals(groups, beat):
    lo = 0
    hi = len(groups) - 1

    while hi != lo:
        mid = math.ceil(lo + (hi - lo) / 2)
        if groups[mid][0].end == beat:
            return mid
        
        if groups[mid][0].end > beat:
            hi = mid - 1
        else:
            lo = mid
    
    return lo


def preprocess_all_songs():
    guarentee_path_exists(NOTE_VECTOR_DIR)
    all_song_paths = glob(f'{SONGS_DIR}*.csv')
    for song_path in tqdm(all_song_paths):
        note_vectors = form_note_vectors(song_path)
        song_number = os.path.basename(song_path).split('.')[0]
        np.save(f'{NOTE_VECTOR_DIR}{song_number}.npy', note_vectors)

if __name__ == "__main__":
    preprocess_all_songs()
    #note_vectors = form_note_vectors(1727)
    #print(len(note_vectors))