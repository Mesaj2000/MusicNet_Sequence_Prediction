import os


DEBUG = True

DEFAULT_K = 30
DEFUALT_SCALE = 4

HIGHEST_NOTE = 104
LOWESET_NOTE = 21
NOTE_RANGE = HIGHEST_NOTE - LOWESET_NOTE + 1

HIGHEST_PREDICTED_NOTE = 45 # Don't ever put higher than 50
LOWEST_PREDICTED_NOTE = 20 # Don't ever put lower than 10
NOTE_PREDICTION_RANGE = HIGHEST_PREDICTED_NOTE - LOWEST_PREDICTED_NOTE + 1

NUM_COMPOSER_CLASSES = 6

TESTING_RATIO = 6 # Really 1/6

SONGS_DIR = 'project/songs/'
TEST_DIR = 'project/test/'
TRAIN_DIR = 'project/train/'
IMAGE_DIR = 'project/output_images/'
SEQUENCE_DIR = 'project/saved_sequences/'
LABEL_DIR = 'project/sequence_labels/'
METADATA_PATH = 'project/musicnet_metadata.csv'
NOTE_VECTOR_DIR = 'project/note_vectors/'

TRAINING_OUTPUT_DIR = 'project/training_out/'
MODELS_DIR = 'project/best_models/'

def debug(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

"""
Returns True if it already existed, false otherwise
"""
def guarentee_path_exists(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        return False
    return True



def get_song_path(song_number):
    return f'{SONGS_DIR}{song_number}.csv'