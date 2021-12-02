import numpy as np
import pandas as pd
import cv2
import os
from intervaltree import Interval, IntervalTree
from pprint import pprint
from util import *
#import preprocessing as pp

class Note_Sequence():
    def __init__(self, notes: list[Interval], start_beat, k, song_number):
        self.song_number = song_number
        self.notes = notes

        if notes is None or len(notes) == 0:
            self.notes = None
            return

        #self.notes = self.scale_notes(scale)

        self.start_beat = start_beat
        self.end_beat = start_beat + k
        self.num_beats = k
        self.name = f'{self.start_beat}-{self.end_beat}'
        
        self.matrix, self.label = self.create_note_matrix_and_label_vector()


    def create_note_matrix_and_label_vector(self) -> tuple[np.ndarray]:
        matrix = np.zeros((NOTE_RANGE, self.num_beats + 1), dtype=int)

        for note in self.notes:
            start = note.begin - self.start_beat
            end = note.end - self.start_beat
            sustained_into_sequence = start < 0

            matrix[note.data, max(start, 0):end] = 0.5
            if not sustained_into_sequence:
                matrix[note.data, start] = 1.0
        
        label = matrix[:,-1]
        matrix = matrix[:,:-1]

        return matrix, label

    
    def save_as_image(self):
        im = self.matrix * 255
        im = np.flip(im, axis=0)

        lab = self.label * 255
        lab = np.flip(lab, axis=0)
        
        path = f'{IMAGE_DIR}{self.song_number}/{self.name}.bmp'
        guarentee_path_exists(path)

        label_path = f'{IMAGE_DIR}{self.song_number}/{self.name}_label.bmp'
        guarentee_path_exists(path)

        cv2.imwrite(path, im)
        cv2.imwrite(label_path, lab)


    def save(self):
        matrix_path = f'{SEQUENCE_DIR}{self.song_number}/{self.name}'
        guarentee_path_exists(matrix_path)

        label_path = f'{LABEL_DIR}{self.song_number}/{self.name}'
        guarentee_path_exists(label_path)

        np.save(matrix_path, self.matrix)
        np.save(label_path, self.label)
        

    # Scales all values by the scale factor
    # Also makes all note beat values integers
    # Also adjusts the note value to fit the window
    def scale_notes(self, scale_factor: int = 1) -> list[Interval]:
        scaled_notes = [Interval(int(note.begin * scale_factor), int(note.end * scale_factor), note.data - LOWESET_NOTE) 
                        for note in self.notes]
        return scaled_notes

    # Shift all data from being reletive to the song to being relative to the sequence
    def make_notes_relative_to_sequence(self) -> list[Interval]:
        start_beat = self.notes[0].begin
        min_note = min(self.notes, key=lambda x: x.data).data
        relative_notes = [Interval(note.begin - start_beat, note.end - start_beat, note.data - min_note)
                          for note in self.notes]
        return relative_notes

    def is_empty(self):
        return self.notes is None
