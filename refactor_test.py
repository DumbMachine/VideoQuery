'''
Scripts for testing the refactoring of the code
'''

import os
import random

from glob import glob

from utils import FRAME_ANNOTATION_PATH, key_frames_parallel

files = glob(FRAME_ANNOTATION_PATH+"/*.pkl")
file = random.choice(files)

# checking if the extraction of key_frames is working as required
key_frames_parallel(path=file)