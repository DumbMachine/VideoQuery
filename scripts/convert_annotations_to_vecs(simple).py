import os
from tqdm import tqdm
from glob import glob

from utils import key_frames_parallel,FRAME_ANNOTATION_PATH, FRAME_NUMBER_PATH, FRAME_VECTORS_PATH

annotations = glob(FRAME_ANNOTATION_PATH+"/*")
dones = 0
notdones = 0
with tqdm(total=len(annotations)) as progress:
    for annotation in annotations:
        # skipping if the files already exists
        filename = annotation.split("/")[-1].split(".")[0]
        vec_path = FRAME_VECTORS_PATH + f"/{filename}.pkl"
        frame_nos_path = FRAME_NUMBER_PATH + f"/{filename}.pkl"

        if os.path.isfile(vec_path) and os.path.isfile(frame_nos_path):
            progress.set_description("Skipping as the file was already done")

        else:
            progress.set_description("Doing a file now")
            key_frames_parallel(annotation)
        progress.update(1)