# Import order matters, wrong order causes invalid poitenr error (cv2)
import os
import cv2
import ngtpy

import pickle
import random

import numpy as np

from glob import glob

import tensorflow as tf
import pathlib

from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as vis_util

annotations = sorted(glob("*annotations.pkl"))
frame_indexes = sorted(glob("*-FRAME-INDEXES.pkl"))
vecs = sorted(glob("*-VECS.pkl"))


def get_single_inference(image, model):
    '''
    Used to get single inferences, mostly used when testing querying
    '''
    input_tensor = tf.convert_to_tensor(
        np.asarray(image.reshape(-1, *image.shape))
    )
    prediction =  model(input_tensor)
    output_dict = {
        "detection_classes": prediction["detection_classes"].numpy(),
        "detection_scores" : prediction["detection_scores"].numpy(),
        "detection_boxes"  : prediction["detection_boxes"].numpy()
    }

    information = []
    for cat, bbox, score in zip(
        output_dict['detection_classes'][0],
        output_dict['detection_boxes'][0],
        output_dict['detection_scores'][0]
    ):
            if score > 0.6:

                # something = [get_index_from_category(cat, self.category_index), score]
                something = [get_index_from_category(cat, category_index), score]

                for box in bbox:
                    something.append(box)

                information.append(something)

    return information

def load_model(
    model_name="ssd_mobilenet_v1_coco_2018_01_28",
    path=os.path.expanduser("~")
):
    base_url = 'http://download.tensorflow.org/models/object_detection/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(
        fname=model_name,
        origin=base_url + model_file,
        untar=True)

    model_dir = pathlib.Path(model_dir)/"saved_model"

    model = tf.saved_model.load(str(model_dir))
    model = model.signatures['serving_default']

    return model

def read_video(path, progress=True):
    '''
    Given the path, reads the video and returns the frames from the video
    '''
    frames = []

    video_reel = cv2.VideoCapture(path)
    fnos = int(video_reel.get(cv2.CAP_PROP_FRAME_COUNT))
    succ , frame = video_reel.read()
    if progress:
        with tqdm(total=fnos) as progress:
            progress.set_description("Reading the Video")
            while succ:
                frames.append(frame)
                succ, frame = video_reel.read()
                progress.update(1)
    else:
        while succ:
            frames.append(frame)
            succ, frame = video_reel.read()

    return frames


def get_index_from_category(category, category_index):
    '''
    Function to return the inverse mapping from category_index (the dict with all the category names)
    '''
    for key in category_index.keys():
        if category_index[key]['name'] == category or key == category:
            return key

category_index = pickle.load(open(os.path.join
            (os.path.expanduser("~"), "youtube", "category_index.pkl"), "rb"
            )
        )


'''
✔️ Initializing the indexes ✔️
'''
dims = 6
max_dims = 10

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

for vecs_path, frame_idxs_path in zip(vecs, frame_indexes):
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

'''
 Searching
'''
model = load_model()

positives = 0

for i in range(100):

    # Taking a random query video
    query_video = random.choice(glob("*.mp4"))
    frames = read_video(query_video)
    # take a random index and read 1000 franes frin ut
    query_start_indx = random.randint(0, len(frames)-1000)
    query_clip = frames[query_start_indx: query_start_indx+1000]
    # choosing uniform frames
    query_frames = np.random.uniform(query_clip, 25)
    query_frames = random.sample(query_clip, 10)

    # FIXME: Instead of single inference, infer all 10 at once
    predictions = []
    for frame in query_frames:
        qvec = np.asarray(
            get_single_inference(frame, model)
        ).flatten()

        if len(qvec):
            result = indexes[len(qvec)].search(qvec, 3)

            for idx, distance in result:
                for key in mapping.keys():
                    for item in mapping[key]:
                        if f"{len(qvec)}-{idx}" in item:
                            predictions.append(key)

    # metrics:
    output = {}
    for pred in predictions:
        if pred in output.keys():
            output[pred] += 1
        else:
            output[pred] = 1

    output = {k: v for k, v in sorted(output.items(), key=lambda item: item[1], reverse=True)}


    query_video, output

    for item in list(output.keys())[:5]:
        if item[:-len(".mp4-annotations.pkl-VECS.pkl")] == query_video.replace(".mp4.mp4", ""):
            positives+=1

# Searching
# FIXME: Reverse the mapping, so now we can search the "x-y-z" string
