import os
import cv2
import pickle

import functools
import operator
from joblib import Parallel, delayed

import numpy as np

from glob import glob
from tqdm import tqdm

from joblib import Parallel, delayed
from utils import CATEGORY_INDEX

frame_idxs_path = sorted(
    glob(".data/results/*key_frame_indexes.pkl"), key=lambda x: x.split("/")[0])[:]
frame_reps_path = sorted(
    glob(".data/results/*key_frame_reps.pkl"), key=lambda x: x.split("/")[0])[:]
videos = sorted(glob(".data/ucf101/**/*.avi"),
                key=lambda x: x.split("/")[0])[:]
category_index = pickle.load(open(CATEGORY_INDEX, "rb"))

dims = 12
objs = []
for a, b, c in zip(frame_idxs_path, frame_reps_path, videos):
    filename = c.split("/")[-1].split(".")[0]
    vec_path  = f".data/results/{filename}.avi-key_frame_reps.pkl-key_frame_reps.pkl"
    fnos_path = f".data/results/{filename}.avi-key_frame_indexes.pkl-key_frame_indexes.pkl"
    if os.path.isfile(vec_path) and os.path.isfile(fnos_path):
        # then its all good
        objs.append([c, vec_path, fnos_path])


def process_file(ob):
    '''

    '''
    vpath, vecs_path, frame_idxs_path = ob
    vecs = pickle.load(open(vecs_path, "rb"))
    frame_idxs = pickle.load(open(frame_idxs_path, "rb"))
    filename = vecs_path.replace("key_frame_reps.pkl", f"frame_reps-{dims}")

    if os.path.isfile(filename):
        return

    vecs_hist = []
    if vecs and frame_idxs:
        for vec, idx in zip(vecs, frame_idxs):
            # TODO: Check for error here, if None is returned from hist
            hist = get_hist(dims, idx, vpath).reshape(dims)
            vec = np.append(vec, hist)
            vecs_hist.append(
                vec
            )

        pickle.dump(
            vecs_hist, open(filename, "wb")
        )
    print(vpath)

# arg_instances = [*zip(frame_idxs_path, frame_reps_path,
#                       range(len(frame_idxs_path)))]


results = Parallel(n_jobs=-1, verbose=1,
                   backend="threading")(map(delayed(process_file), objs))


# Total things done
total = sorted(
    glob(f"*/key_frame_reps-{dims}.pkl"), key=lambda x: x.split("/")[0])[:]


def get_hist(dimension, frame_idx, vpath, gray=True):
    '''
    Util function when we wanna add in the histogram
    @parameters:
    ------------
    dimensions: bins for the histogram
    frame_idx: frame idx in the video
    vpath: the path for the video file
    '''
    video_reel = cv2.VideoCapture(vpath)
    video_reel.set(1, frame_idx)
    suc, image = video_reel.read()
    if suc:
        if gray:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        hist = cv2.calcHist([image], [0], None, [dimension], [0, 256])
        return hist
    else:
        return None

# with tqdm(total=len(frame_reps_path)) as progress:
#     for vecs_path, frame_idxs_path in zip(frame_reps_path, frame_idxs_path):

#         try:

#             vecs = pickle.load(open(vecs_path, "rb"))
#             frame_idxs = pickle.load(open(frame_idxs_path, "rb"))
#             vpath = vecs_path.replace("/key_frame_reps.pkl", ".avi")
#             filename = vecs_path.replace("frame_reps", f"frame_reps-{dims}")

#             if os.path.isfile(filename):
#                 progress.update(1)
#                 continue

#             if vecs and frame_idxs:
#                 progress.set_description(filename)
#                 for vec, idx in zip(vecs, frame_idxs):
#                     # TODO: Check for error here, if None is returned from hist
#                     hist = get_hist(dims, idx, vpath).reshape(dims)
#                     vec = np.append(vec, hist)

#                     pickle.dump(
#                         vec, open(filename, "wb")
#                     )
#                 progress.update(1)

#         except Exception as e:
#             print(e)
#             progress.update(1)
