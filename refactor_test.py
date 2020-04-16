'''
Scripts for testing the refactoring of the code
'''

import os
import random

from glob import glob

from utils import FRAME_VIDEO_PATH

files = glob(FRAME_VIDEO_PATH+"/*.avi")
file = random.choice(files)

filename = file