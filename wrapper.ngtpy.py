import itertools
import os
import pathlib
import pickle
import random
import sys
import tarfile
import urllib.request
import uuid
import zipfile
from collections import Counter, defaultdict
from glob import glob
from io import StringIO
from pathlib import Path

import cv2
import imageio
import matplotlib.pyplot as plt
import ngtpy
import numpy as np
import pafy
import pandas as pd
import six.moves.urllib as urllib
import tensorflow as tf
from IPython.display import display
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
from tqdm import tqdm

dims = 12
'''
Making the dataset:
It will have the following structure:
- DATA[6]: will have all the rows, with len = 6
- ............................................ (12)
...................................................
...................................................
- DATA[30]

'''
'''
# TODO: Errors
- Empty index
'''
'''
To run the code:
engine = VideoEngine('/home/ratkum/friends/friends')
engine.build_indexes()
engine.build_dataset()
engine.insert_dataset()
'''

class VideoEngine:
    '''
    Class Object used to create the video search engine
    '''
    def __init__(self, fnos_directory, vecs_directory):
        '''
        Some string
        '''
        self.DIMS = 6
        self.MAX_INDEXES = 40

        self.fnos_directory = fnos_directory
        self.vecs_directory = vecs_directory
        self.mapping = {}
        self.INDX = {}
        self.DATA = {}

        self.predictions = []
        self.distances = {}

    def build_dataset(self):
        '''
        Using the directory provided,we will read the row vector
        and store them in a dict of the appropriate site
        '''
        print("building dataset")
        # for path in glob("*frame_vectors.pkl"):
        for frame_idxs_path, vecs_path in zip(
            sorted(glob(self.fnos_directory + "/*.pkl")), sorted(glob(self.vecs_directory + f"/*.pkl"))):
        # for path in directory.glob("/*frame_vectors.pkl"):
            vecs = pickle.load(open(vecs_path, "rb"))
            frame_idxs = pickle.load(open(frame_idxs_path, "rb"))
            # self.mapping[str(path).split('/')[-2]] = []
            if vecs and frame_idxs:
                self.mapping[vecs_path] = []
                print(vecs_path)
                for vec, frame_idx in zip(vecs, frame_idxs):
                    # Each vec object will have 2 frames and thus 2 frame vectors
                    # try:
                    self.DATA[len(vec)].append(vec)
                # self.mapping[str(path).split('/')[-2]].append(
                    self.mapping[vecs_path].append(
                        # Format: <INDX_dimension>-<idx_in_its_INDX>
                        # f"{len(vec)}-{len(self.DATA[len(vec)])}-{frame_idx_in_source}"
                        f"{len(vec)}-{len(self.DATA[len(vec)])}-{frame_idx}" # Reason for -1 is that, the indexing starts at zero
                    )
                    # except Exception as e:
                    #     print(e)

    def build_indexes(self):
        '''
        Will initialize the indexes
        '''
        for i in range(self.MAX_INDEXES):
            tempfile = str.encode(f"{(i+1)}-Index-tmp")
            ngtpy.create(tempfile, self.DIMS*(i+1))
            self.INDX[self.DIMS*(i+1)] = ngtpy.Index(tempfile)

            self.DATA[self.DIMS*(i+1)] = []

    def insert_dataset(self):
        '''
        The data collected in self.DATA would be inserted in the appropriate INDX
        '''
        for key in self.DATA.keys():
            # Inserting NULL data gives error
            if self.DATA[key]:
                self.INDX[key].batch_insert(self.DATA[key])
                self.INDX[key].save()

        for key in engine.DATA.keys():
            # Inserting NULL data gives error
            if engine.DATA[key]:
                engine.INDX[key].batch_insert(engine.DATA[key])
                engine.INDX[key].save()

    def _get_image_samples(self, query_clip):
        '''
        @parameters:
        ------------
        - query_clip: The Video path to the query image
        '''

        # Sampling the 5 frames from the query_clip frames
        vid = cv2.VideoCapture(query_clip)
        # Generating samples between a certain 0 to end_frame ( to emulate the querying a of a part of the whole video)
        start_frame = random.randint(0, int(vid.get(cv2.CAP_PROP_FRAME_COUNT) * 0.9))
        end_frame = random.randint(int(vid.get(cv2.CAP_PROP_FRAME_COUNT) * 0.9), int(vid.get(cv2.CAP_PROP_FRAME_COUNT)))
        rfnos = [
            int(number) for number in np.random.uniform(
                start_frame, end_frame,
                int(vid.get(cv2.CAP_PROP_FPS))
            )
        ]

        frames = []
        for fno in rfnos:
            vid.set(1, fno)
            _, frame = vid.read()
            frames.append(frame)
        return frames

    def _search(self, query, neighbours=3):
        '''
        Helper function used to search the proper index
        '''
        result = self.INDX[len(query)].search(
            query, neighbours
        )

        predictions = []
        distances = {}

        for idx, distance in result:
            origin_video = self.get_mapping(len(query), idx)
            if origin_video:
                predictions.append(origin_video)
                if origin_video in distances.keys():
                    distances[origin_video] += distance
                else:
                    distances[origin_video] = distance
        return predictions, distances

    # TODO: Return when None predictions
    # FIXME: Class a function to clean the state variables of distances and predictions
    def search_clip(self, query_clip, model):
        '''
        @parameters:
        ------------
        - query_clip: The Video path to the query image
        '''
        # Searching for results of each image from the sample
        sample_frames = self._get_image_samples(query_clip)
        for frame in sample_frames:
            query_row = np.asarray(
                get_single_inference(frame, model)
            ).flatten()
            if len(query_row):
                prediction, distance = self._search(query=query_row)
                self.predictions += [*prediction]
                self.distances.update(distance)
            # query_information = np.asarray(
            #     get_single_inference(frame, model)
            # ).flatten()

            # # length comes out as zero, if no object was found in the image
            # if len(query_information):
            #     result = self.INDX[len(query_information)].search(
            #         query_information, 3
            #     )

            #     for idx, distance in result:
            #         origin_video = self.get_mapping(len(query_information), idx)
            #         if origin_video is not None:
            #             predictions.append(origin_video)

            #             if origin_video in distances.keys():
            #                 distances[origin_video] += distance
            #             else:
            #                 distances[origin_video] = distance

        self.output = {}
        for pred in set(self.predictions):
            self.output[pred] = self.predictions.count(pred)

        self.output = {k: v for k, v in sorted(self.output.items(), key=lambda item: item[1], reverse=True)}
        self.distances = {k: v for k, v in sorted(self.distances.items(), key=lambda item: item[1])}

        for key in self.output.keys():
            self.output[key] = (
                self.output[key],
                self.distances[key]
            )

        # Sorting the predictions on basis of 2 rules:
        # - More predicted times frame first
        # - Shorter distance first, among same times predicted
        self.temp = sorted(
                list(
                    self.output.items()
                ),
                key= lambda x: x[1][0]
            )

    def search_vectors(self, query_row=None, query_rows=None):
        """Search for a vector in the instance.INDXS
        """
        predictions = []
        distances = {}
        if query_row is not None:
            prediction, distance = self._search(query=query_row)
            predictions += [*prediction]
            distances.update(distance)
        elif query_rows is not None:
            for query_row in query_rows:
                prediction, distance = self._search(query=query_row)
                predictions += [*prediction]
                distances.update(distance)
        else:
            raise Exception("What the hell man")

        self.output = {}
        for pred in set(predictions):
            self.output[pred] = predictions.count(pred)

        self.output = {k: v for k, v in sorted(self.output.items(), key=lambda item: item[1], reverse=True)}
        self.distances = {k: v for k, v in sorted(distances.items(), key=lambda item: item[1])}

        for key in self.output.keys():
            self.output[key] = (
                self.output[key],
                distances[key]
            )

        # Sorting the predictions on basis of 2 rules:
        # - More predicted times frame first
        # - Shorter distance first, among same times predicted
        self.temp = sorted(
                list(
                    self.output.items()
                ),
                key= lambda x: x[1][0]
            )


    # TODO: Save this mapping in data base: Search by qlen and idx then obtain frame_number
    # FIXME: Fix the None case
    def get_mapping(self, qlen, idx):
        '''
        Returns the mapping of the key to the origin url
        '''
        for key in self.mapping.keys():
            for val in self.mapping[key]:
                if f"{qlen}-{idx}" in val:
                    return key
        return None



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


def get_index_from_category(category, category_index):
    '''
    Function to return the inverse mapping from category_index (the dict with all the category names)
    '''
    for key in category_index.keys():
        if category_index[key]['name'] == category or key == category:
            return key

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



engine = VideoEngine(
    fnos_directory="/home/dumbmachine/code/SVMWSN/.data/frame_numbers",
    vecs_directory="/home/dumbmachine/code/SVMWSN/.data/vectors-18"
)
engine.build_indexes()
engine.build_dataset()
engine.insert_dataset()


def test_something():
    plus = 0
    query_clips = random.sample([_ath for _ath in glob("/home/dumbmachine/code/SVMWSN/.data/vectors-18/*")], 1000)
    for intr, query_clip in enumerate(query_clips):
        print(intr, len(query_clips))
        vectors = pickle.load(open(query_clip, "rb"))
        if len(vectors):
            engine.search_vectors(query_row=vectors[0])
            predictions = [i[0] for i in engine.temp]
            if query_clip in predictions:
                plus+=1

    print(plus/len(query_clips))
