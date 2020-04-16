"""
This is the place where we will test the code
"""
from utils import get_key_from_batch, compare_segments
from sklearn.cluster import KMeans
from collections import Counter
import pandas as pd
import numpy as np
import pickle
from utils import SAMPLE_VIDEO
from frame_loader import FrameLoader
loader = FrameLoader(path=SAMPLE_VIDEO)

# loader.build_annotations()


# testing if the code actually works now
frames = loader._read_video()

# The code below if for testing the key frames generation


frames = loader._read_video()
annotations = pickle.load(
    open(".data/sample_sample.mp4-annotations.pkl", "rb"))
category_index = pickle.load(open(".data/category_index.pkl", "rb"))


frame_nos = []
frame_vecs = []
start = 0
for end in range(30, len(annotations['frames']), 30):
    print(start, end)
    try:
        # compare_segments(batch=annotations['frames'][start:end],
        #                  frames=frames[start:end],
        #                  frame_size=3, save=True, filename=f"{start}-{end}.mp4")
        temp = get_key_from_batch_with_frame_nos(
            batch=annotations['frames'][start:end],
            start=start, category_index=category_index
        )
        frame_nos.extend(temp[0])
        frame_vecs.extend(temp[1])
        print(len(temp[0]), len(temp[0]))
    except Exception as e:
        print(f"{start}-{end}-{e}")

    start = end
    print("===========================================")

pickle.dump(frame_nos, )