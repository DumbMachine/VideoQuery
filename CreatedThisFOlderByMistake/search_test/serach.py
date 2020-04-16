import imageio
import matplotlib.pyplot as plt
# Import order matters, wrong order causes invalid poitenr error (cv2)
import os
import cv2
import ngtpy
import pickle
import random

import numpy as np

from glob import glob
from tqdm import tqdm
import tensorflow as tf
import pathlib



videos = sorted(glob("*/*.mp4"), key=lambda x: x.split("/")[0])
frame_idxs = sorted(glob("*/key_frame_indexes.pkl"), key=lambda x: x.split("/")[0])
frame_reps = sorted(glob("*/key_frame_reps.pkl"), key=lambda x: x.split("/")[0])

for idxs_path, reps_path in zip(frame_idxs, frame_reps):
    idxs = pickle.load(open(idxs_path, "rb"))
    reps = pickle.load(open(reps_path, "rb"))

    break

category_index = pickle.load(open(os.path.join(os.path.expanduser("~"), "youtube", "category_index.pkl"), "rb"))



'''
✔️ Initializing the indexes ✔️
'''
dims = 6
max_dims = 20

indexes = {}
data = {}

for i in range(max_dims):
    tempfile = str.encode(f"{(i+1)}-Index-tmp")
    ngtpy.create(tempfile, dims*(i+1))
    indexes[dims*(i+1)] = ngtpy.Index(tempfile)

    data[dims*(i+1)] = []

'''
- Reading the vectors and adding them to the appropriate index_size data object,
- Making the mapping
'''
# Format: <file_name> : ["<len(vec)>-<loc_of_vec_in_index>-<frame_number_in_video>"]
mapping = {}

for vecs_path, frame_idxs_path in zip(frame_reps, frame_idxs):
    _vecs = pickle.load(open(vecs_path, "rb"))
    _frame_idxs = pickle.load(open(frame_idxs_path, "rb"))

    if _vecs and _frame_idxs:
        mapping[vecs_path] = []
        for vec, idx in zip(
            _vecs,
            _frame_idxs
        ):
            data[len(vec)].append(vec)
            mapping[vecs_path].append(
                f"{len(vec)}-{len(data[len(vec)])}-{idx}"
            )

# Adding the data to indexers
for key in data.keys():
    if data[key]:
        indexes[key].batch_insert(data[key])

model = load_model()

positives = 0
iterations = 100
for i in range(iterations):
    print(i)
    query_video = random.choice(frame_reps).replace("/key_frame_reps.pkl", ".avi")
    frames = read_video(query_video)
    # frames = read_video("v_WritingOnBoard_g03_c06.avi")
    # take a random index and read 1000 franes frin ut
    # query_start_indx = random.randint(0, int(len(frames)*0.5))
    query_start_indx = len(frames)
    # query_clip = frames[query_start_indx: int(query_start_indx+len(frames)*0.2)]
    # choosing uniform frames
    # query_frames = np.random.uniform(query_clip, 10)
    # query_frames = np.random.uniform(query_clip, 10)
    # if len(query_clip) < 10:
        # continue
    # query_frames = random.sample(query_clip, 10)
    query_frames = frames[:int(0.2*len(frames))]


    predictions = []
    for frame in query_frames:
        qvec = np.asarray(
            get_single_inference(frame, model, category_index)
        ).flatten()

        if len(qvec):
            result = indexes[len(qvec)].search(qvec, 3)

            for idx, distance in result:
                for key in mapping.keys():
                    for item in mapping[key]:
                        if f"{len(qvec)}-{idx}" == "-".join(item.split("-")[:-1]):
                            predictions.append(key.split("/")[0])

    # metrics:
    output = {}
    for pred in predictions:
        if pred in output.keys():
            output[pred] += 1
        else:
            output[pred] = 1

    output = {k: v for k, v in sorted(output.items(), key=lambda item: item[1], reverse=True)}

    query_video, output

    for item in list(output.keys())[:3]:
        if item == query_video.replace(".avi", ""):
            positives+=1

positives/iterations




def visualize_search(frame, name):
    '''
    input is the image frame/picture
    '''
    image_array = []
    video = []
    qvec = np.asarray(
        get_single_inference(frame, model, category_index)
    ).flatten()

    if len(qvec):
        result = indexes[len(qvec)].search(qvec, 3)

        for idx, distance in result:
            for key in mapping.keys():
                for item in mapping[key]:
                    if f"{len(qvec)}-{idx}" == "-".join(item.split("-")[:-1]):
                        print(item, idx)
                        image_array.append(
                            (item.split("-")[-1], key.replace("/key_frame_reps.pkl", ".avi"))
                        )

    # FIXME: Sometimes this doesnt return proper results
    images = [frame]
    for idx, video_name in image_array:
        video_reel = cv2.VideoCapture(video_name)
        video_reel.set(1, int(idx))
        suc, image = video_reel.read()
        if suc:
            images.append(image)

    if len(images) > 1:
        max_rows, max_cols = 2, 2
        fig, axes = plt.subplots(nrows=max_rows, ncols=max_cols, figsize=(20,20))
        for idx, image in enumerate(images):
            row = idx // max_cols
            col = idx % max_cols
            axes[row, col].axis("off")
            if idx == 0:
                axes[row, col].set_title("query-frame", fontsize=50)
            else:
                axes[row, col].set_title(f"result-{idx}-frame", fontsize=30)

            axes[row, col].imshow(image, cmap="gray", aspect="auto")

        plt.subplots_adjust(wspace=.05, hspace=.05)
        fig.canvas.draw()
        video = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plt.savefig(f"figures/{name}.jpeg")
        return video.reshape(fig.canvas.get_width_height()[::-1] + (3,))

'''
draw test
'''
query_video = random.choice(frame_reps).replace("/key_frame_reps.pkl", ".avi")
frames = read_video(query_video)

# taking only 10 frames
query_start_indx = random.randint(0, int(len(frames)*0.5))
query_clip = frames[query_start_indx: int(query_start_indx+len(frames)*0.2)]

images = []
video = []
if len(query_clip) > 10:
    query_frames = frames[:int(0.2*len(frames))]
    for itr, frame in enumerate(query_frames):
        video.append(visualize_search(frame, itr))

imageio.mimsave("figures/query.mp4", [frame for frame in video if frame])

