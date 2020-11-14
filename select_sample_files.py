"""
Select sample files from library

Run this and direct output to file, e.g., selected_files.txt

C. Cafiero
"""
import os


ROOT_DIR = '/Volumes/Public/Music/Library/Jazz/Tenor Booster Packs/BP4'


if __name__ == '__main__':

    for root, subdirs, files in os.walk(ROOT_DIR):
        for file in files:
            trackname, ext = os.path.splitext(file)
            if ext in ['.flac', '.mp3']:
                if os.path.isfile(os.path.join(root, file)):
                    print(os.path.join(root, file))
