
import os
from tqdm import tqdm
from glob import glob
from joblib import Parallel, delayed

from utils import key_frames_parallel, save_vec_with_histograms, FRAME_ANNOTATION_PATH, FRAME_NUMBER_PATH, FRAME_VECTORS_PATH

"""
Will convert the annotation to the required naive vectors
"""

# annotations = glob(FRAME_ANNOTATION_PATH+"/*")
# with tqdm(total=len(annotations)) as progress:
#     for annotation in annotations:
#         # skipping if the files already exists
#         # filename = annotation.split("/")[-1].split(".")[0]+".avi"
#         progress.set_description(annotation)
#         key_frames_parallel(annotation)
#         progress.update(1)

"""
Will add the histogram details for the vectors
"""


annotations = glob(FRAME_VECTORS_PATH+"/*")
filenames = [annotation.split("/")[-1].split(".")[0]+".avi" for annotation in annotations]
results = Parallel(n_jobs=-1, verbose=0, backend="loky")(
        tqdm(map(delayed(save_vec_with_histograms), filenames), total=len(filenames)))