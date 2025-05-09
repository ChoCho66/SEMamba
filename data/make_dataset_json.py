import os
import json
import argparse

def list_files_in_directory(directory_path):
    # List all files in the directory
    files = []
    for root, dirs, filenames in os.walk(directory_path):
        for filename in filenames:
            if filename.endswith('.wav'):   # only add .wav files
                files.append(os.path.join(root, filename))
    return files

def save_files_to_json(files, output_file):
    with open(output_file, 'w') as json_file:
        json.dump(files, json_file, indent=4)

def make_json(directory_path, output_file):
    # Get the list of files and save to JSON
    files = list_files_in_directory(directory_path)
    save_files_to_json(files, output_file)

# create training set json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix_path', default='None')

    args = parser.parse_args()

    # prefix = args.prefix_path if (args.prefix_path is not None) else "/mnt/e/Corpora/noisy_vctk/"
    prefix = "/disk4/chocho/_datas/VCTK_DEMAND16k"

    # prefix = args.prefix_path if (args.prefix_path is not None) else "/home/kycho/Speech-Enhancement-Code/datas/VCTK-Valentini-Botinhao-2016-16k-1"
    # prefix = "/home/kycho/Speech-Enhancement-Code/datas/VCTK-Valentini-Botinhao-2016-16k-1"

    ## You can manualy modify the clean and noisy path
    # ----------------------------------------------------------#
    ## train_clean
    make_json(
        os.path.join(prefix, 'train/clean'),
        'data/train_clean.json'
        # os.path.join(prefix, 'clean_train'),
        # 'train_clean.json'
    )

    ## train_noisy
    make_json(
        os.path.join(prefix, 'train/noisy'),
        'data/train_noisy.json'
        # os.path.join(prefix, 'noisy_train/'),
        # 'train_noisy.json'
    )
    # ----------------------------------------------------------#

    # ----------------------------------------------------------#
    # create valid set json
    ## valid_clean
    make_json(
         os.path.join(prefix, 'test/clean'),
        'data/valid_clean.json'
        #  os.path.join(prefix, 'clean_test/'),
        # 'valid_clean.json'
    )

    ## valid_noisy
    make_json(
        os.path.join(prefix, 'test/noisy'),
        'data/valid_noisy.json'
        # os.path.join(prefix, 'noisy_test/'),
        # 'valid_noisy.json'
    )
    # ----------------------------------------------------------#

    # ----------------------------------------------------------#
    # create testing set json
    ## test_clean
    # make_json(
    #    os.path.join(prefix, 'clean_testset_wav_16k/'),
    #     'data/test_clean.json'
    # )

    # ## test_noisy
    # make_json(
    #    os.path.join(prefix, 'noisy_testset_wav_16k/'),
    #     'data/test_noisy.json'
    # )
    # ----------------------------------------------------------#


if __name__ == '__main__':
    main()
