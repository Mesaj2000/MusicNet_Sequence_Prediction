import os
import shutil
from glob import glob
import pandas as pd
from util import TRAINING_OUTPUT_DIR, MODELS_DIR
from train_model import filename_generator



def get_best_model(filename):
    all_csvs = glob(f'{TRAINING_OUTPUT_DIR}/2021-11-30_20-52-21/{filename}.csv')
    best_acc = 0
    best_csv = ''
    for csv in all_csvs:
        df = pd.read_csv(csv)
        acc = df['Test Accuracy'].max()
        if acc > best_acc:
            best_acc = acc
            best_csv = csv

    print(best_acc, best_csv)
    return best_csv.replace('.csv', '.pt')


def main():
    model_paths = []

    filename = filename_generator(False, None)
    model_paths.append(get_best_model(filename))

    for composer in '012345*':
        filename = filename_generator(True, composer)
        model_paths.append(get_best_model(filename))


    for model_path in model_paths:
        #print(model_path)
        pass
        shutil.copy(model_path, MODELS_DIR)

if __name__ == "__main__":
    #main()
    pass
