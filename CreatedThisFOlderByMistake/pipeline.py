import itertools
import os
import pathlib
import pickle
import random
import sys
import tarfile
import urllib.request
import uuid
import warnings
import zipfile
from collections import Counter, defaultdict
from io import StringIO
from pathlib import Path

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import six.moves.urllib as urllib
import tensorflow as tf
from IPython.display import display
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
from tqdm import tqdm

import pafy
from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as vis_util

warnings.filterwarnings("ignore")
'''
# TODO: Check if file already exists

'''


class Image2Binary:
    def __init__(
            self,
            url=None,
            model=None,
            video=None,
            category_index=None,
            video_path=None,
            save_directory=None):
        '''
        [ USAGE ]:

        model = load_model()
        video = Image2Binary(url=url, model=model, save_directory="videos")
        video._download_video()
        video.create_annotations()
        video.get_fnos()
        video.make_video()
        import time
        time.sleep(1000000)

        video = Image2Binary(video_path="friends/Friends_20-_20S03E20_20-_20The_20One_20With_20The_20Dollhouse.mp4", model=model, save_directory="videos")
        video.create_annotations()
        video.get_fnos()
        # video.convert_annotations_to_bins()
        # video.save_rows()

        Initialized the Object with the required variables
        @parameters:
        ------------
        - url: URL of the youtube video
        - path: path to Video file
        '''
        if url is None and video_path is None:
            raise EnvironmentError

        self.quit = False
        self.url = url
        self.model = model
        self.category_index = pickle.load(open(os.path.join
                                               (os.path.expanduser(
                                                   "~"), "category_index.pkl"), "rb"
                                               )
                                          )
        self.video_path = video_path
        self.save_directory = save_directory

        if save_directory is None:
            self.save_directory = "."
        if not os.path.isdir(self.save_directory):
            os.makedirs(save_directory)

    def _download_video(self, quiet=True):
        '''
        Downloads the video off internet:
            - Currently only supports downloading videos from Youtube at 360p
        '''
        STREAM = "normal:mp4@640"

        video = pafy.new(self.url)
        dl_stream = None
        streams = video.streams
        for stream in streams:
            if STREAM in str(stream):
                dl_stream = stream
                break
        if dl_stream is None:
            raise Exception(
                "The required quality is not abailable for this video")
        self.path = os.path.join(
            self.save_directory, f"{video.title}"
        )
        if os.path.isfile(self.path+"/video.mp4"):
            self.quit = True

        # Writing the video to base_directory/video_name/video.mp4
        if not os.path.isdir(self.path):
            os.makedirs(self.path)
        dl_stream.download(self.path+"/video.mp4", quiet=quiet)

    def _read_video(self, progress=False):
        '''
        Given the path, reads the video and returns the frames from the video
        '''
        if self.quit:
            return
        frames = []
        if self.video_path:
            video_reel = cv2.VideoCapture(self.video_path)
        else:
            video_reel = cv2.VideoCapture(self.path+"/video.mp4")

        self.path = self.video_path[:-4]

        self.fnos = int(video_reel.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(video_reel.get(cv2.CAP_PROP_FPS))
        succ, frame = video_reel.read()
        if progress:
            with tqdm(total=self.fnos) as progress:
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

    def create_annotations(self, progress=False, batch_size=256, save=True):
        '''
        Creates the annotations pickle file which has all the predictions
        '''
        if self.quit:
            return

        self.annotations = {
            "frames": []
        }
        frames = self._read_video(progress=False)
        if progress:
            raise NotImplementedError

        else:
            if len(frames) < batch_size:
                input_tensor = tf.convert_to_tensor(np.asarray(frames))
                # predictions = self.model(input_tensor)
                prediction = self.model(input_tensor)

                for idx in range(len(prediction['detection_scores'])):
                    output_dict = {
                        "detection_classes": prediction["detection_classes"][idx],
                        "detection_scores": prediction["detection_scores"][idx],
                        "detection_boxes": prediction["detection_boxes"][idx].numpy()
                    }
                    self.annotations['frames'].append(
                        {
                            "annotations": fn(output_dict, self.category_index)
                        }
                    )

            else:
                start = 0
                predictions = []

                for end in range(batch_size, len(frames), batch_size):
                    input_tensor = tf.convert_to_tensor(
                        np.asarray(frames[start: end])
                    )
                    # predictions.append(self.model(input_tensor))
                    prediction = self.model(input_tensor)

                    for idx in range(len(prediction['detection_scores'])):
                        output_dict = {
                            "detection_classes": prediction["detection_classes"][idx],
                            "detection_scores": prediction["detection_scores"][idx],
                            "detection_boxes": prediction["detection_boxes"][idx].numpy()
                        }
                        self.annotations['frames'].append(
                            {
                                "annotations": fn(output_dict, self.category_index)
                            }
                        )

                    start = end

            # for prediction in predictions:
            #     for idx in range(len(prediction['detection_scores'])):
            #         output_dict = {
            #             "detection_classes": prediction["detection_classes"][idx],
            #             "detection_scores" : prediction["detection_scores"][idx],
            #             "detection_boxes"  : prediction["detection_boxes"][idx]
            #         }
            #         self.annotations['frames'].append(
            #             {
            #                 "annotations": fn(output_dict, self.category_index)
            #             }
            #         )

            if save:
                pickle.dump(
                    self.annotations,
                    open(
                        f"{self.path}-annotations.pkl", "wb"
                        # f"{self.path}/annotations.pkl", "wb"
                    )
                )

    def get_fnos(self, save=True):
        self.FNOS = []
        start = 0
        for end in range(self.fps, len(self.annotations['frames']), self.fps):
            try:
                batch = self.annotations['frames'][start:end]
                temp = get_key_from_batch(batch, start, self.category_index)
                for _temp in temp:
                    self.FNOS.append(_temp)
            except Exception as e:
                print(f"{start}-{end}-{e}")

            start = end

        start = 0
        for end in range(video.fps, len(video.annotations['frames']), video.fps):
            try:
                batch = video.annotations['frames'][start:end]
                temp = get_key_from_batch(batch, start, video.category_index)
                for _temp in temp:
                    video.FNOS.append(_temp)
            except Exception as e:
                print(f"{start}-{end}-{e}")

            start = end

        # video.FNOS = []
        # start = 0
        # for end in range(video.fps, len(video.annotations['frames']), video.fps):
        #     try:
        #         batch = video.annotations['frames'][start:end]
        #         temp = get_key_from_batch(batch, start, video.category_index)
        #         for _temp in temp:
        #             video.FNOS.append(_temp)
        #     except Exception as e:
        #         print(f"{start}-{end}-{e}")

        #     start = end

        # start = 0
        # for end in range(video.fps, len(video.annotations['frames']), video.fps):
        #     try:
        #         batch = video.annotations['frames'][start:end]
        #         temp = get_key_from_batch(batch, start, video.category_index)
        #         for _temp in temp:
        #             video.FNOS.append(_temp)
        #     except Exception as e:
        #         print(f"{start}-{end}-{e}")

        #     start = end

        # Converting the format to np array
        # video.FNOS = np.asarray(
        #     [np.asarray(i) for i in self.FNOS]
        # )

        self.frames = sorted([int(i[0]) for i in self.FNOS])

        if save:
            pickle.dump(
                self.FNOS,
                open(
                    f"{self.path}-frame_vectors.pkl", "wb"
                    # f"{self.path}/frame_vectors.pkl", "wb"
                )
            )

    # FIXME: Sort before something
    def make_video(self):
        frames = self._read_video(progress=False)
        imageio.mimsave(f'{self.path}-SMALLER-.mp4',
                        [frames[i][0] for i in self.frames])

    def convert_annotations_to_bins(self):
        bins = []
        information = []
        for frame in self.annotations['frames']:
            _information = []
            for cat, bbox, score in zip(
                frame['annotations']['detection_classes'],
                frame['annotations']['detection_boxes'],
                frame['annotations']['detection_scores']
            ):
                    if score > 0.6:

                        something = [get_index_from_category(
                            cat, self.category_index), score]

                        for box in bbox:
                            something.append(box)

                        _information.append(something)
            information.append(_information)

        for fno in self.FNOS:
            bins.append(
                band_array([hash_the_row(
                    elem[0], elem[1], elem[2:]
                ) for elem in information[fno]])
            )

        pickle.dump(
            bins,
            open(
                f"{self.path}-BINARY.pkl", "wb"
            )
        )


def get_single_inference(image, model):
    '''
    Used to get single inferences, mostly used when testing querying
    '''
    input_tensor = tf.convert_to_tensor(
        np.asarray(image.reshape(-1, *image.shape))
    )
    prediction = model(input_tensor)
    output_dict = {
        "detection_classes": prediction["detection_classes"].numpy(),
        "detection_scores": prediction["detection_scores"].numpy(),
        "detection_boxes": prediction["detection_boxes"].numpy()
    }

    information = []
    for cat, bbox, score in zip(
        output_dict['detection_classes'][0],
        output_dict['detection_boxes'][0],
        output_dict['detection_scores'][0]
    ):
            if score > 0.6:

                # something = [get_index_from_category(cat, self.category_index), score]
                something = [get_index_from_category(
                    cat, category_index), score]

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


def get_index_from_category(category, category_index):
    '''
    Function to return the inverse mapping from category_index (the dict with all the category names)
    '''
    for key in category_index.keys():
        if category_index[key]['name'] == category or key == category:
            return key


def fn(output_dict, category_index):
    if np.where(output_dict['detection_scores'] > 0.5)[0].size > 0:
        best = [
            output_dict["detection_scores"][i].numpy()
            for i in np.where(output_dict['detection_scores'] > 0.5)[0]
        ]
        return {
            "detection_classes": [category_index[int(idx)]['name'] for idx in output_dict['detection_classes']][:len(best)],
            "detection_scores": best,
            "detection_boxes": output_dict['detection_boxes'][:len(best)],
        }
    else:
        return {
            "detection_classes":  [],
            "detection_scores":  [],
            "detection_boxes":  []
        }


def search(biglist, smallist):
    for i in range(len(biglist)):
        if smallist in biglist[i]:
            return i, biglist[i].index(smallist)
    return None, None


def hamming2(s1, s2):
    """Calculate the Hamming distance between two bit strings"""
    assert len(s1) == len(s2)
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def band_array(arr):
    '''
    If the array has multiple elements, it'll AND all of them
    '''
    barr = "".join([str(1) for _ in range(len(arr[0]))])
    for elem in arr:
        barr = band(elem, barr)

    return barr


def band(s1, s2=None):
    """Binary addition"""
    if s2 is None:
        s2 = "".join([str(1) for _ in range(len(s1))])
    assert len(s1) == len(s2)
    ret = ""
    for i, j in zip(s1, s2):
        ret += str(1) if i == j else str(0)
    return ret


def bfloat(ffloat):
    if int(ffloat) != ffloat:
        return bin(int("".join([i for i in str(ffloat).split(".")[-1] if i != '0'][:2])))[2:].zfill(8)
    else:
        return bin(int(ffloat)).zfill(8)


def get_key_from_batch(batch, start, category_index):
    '''
    Returns the key frames from the given batch of frames
    '''
    # Getting the required information for each frame
    ## Getting all the annotations with more than 0.6 confidence.
    information = []
    for frame in batch:
        _information = []
        for cat, bbox, score in zip(
            frame['annotations']['detection_classes'],
            frame['annotations']['detection_boxes'],
            frame['annotations']['detection_scores']
        ):
                ## 1. Only bboxes
                # something = []
                ## 2. Bboxes and class
                if score > 0.5:

                    something = [get_index_from_category(
                        cat, category_index), score]

                    for box in bbox:
                        something.append(box)

                    _information.append(something)
        information.append(_information)

    information = [temp for temp in information if temp]
    if not information:
        return []
    max_objs = max([len(information[i]) for i in range(len(information))])

    clustering_data = []
    for i in range(max_objs):
        _data = []
        for j in range(len(information)):
            if len(information[j]) > i:
                _data.append(information[j][i])
        clustering_data.append(_data)

    # The above output implies, we can divide these frame
    # into a max of 4 frames (considering the lowest last object occurance)
    FRAMES_THRESHOLD = 2
    clustering_data_prediction = []
    for i in range(len(clustering_data)):
        if len(clustering_data[i]) >= FRAMES_THRESHOLD:
            kmeans = KMeans(n_clusters=FRAMES_THRESHOLD).fit(
                clustering_data[i])
            clustering_data_prediction.append(kmeans.labels_)

    indx = len(clustering_data)
    for i in range(len(clustering_data)):
        if len(clustering_data[i]) < FRAMES_THRESHOLD:
            indx = i
            break

    data = []
    for frame_nos, info in enumerate(information):
        for inf in info:
            pred_index, index = search(clustering_data[:indx], inf)
            if not pred_index and not index:
                # The case of last three things
                pred_index, index = search(clustering_data, inf)
                something = [start+frame_nos]+inf+[index]
                data.append(something)
            else:
                cluster = clustering_data_prediction[pred_index][index]
                something = [start+frame_nos]+inf+[cluster]
                data.append(something)

    df = pd.DataFrame(data, columns=[
        "frame_nos", "category", "score",
        "bbox1", "bbox2", "bbox3", "bbox4", "cluster"
    ])
    df['avg_cluster'] = None

    for frame in df.frame_nos:
        df.loc[df.frame_nos == frame, "avg_cluster"] = Counter(
            df[df.frame_nos == frame].cluster.values
        ).most_common()[::-1][-1][0]

    fnos = []
    for cluster in range(FRAMES_THRESHOLD):
        # Taking the frame with the most number of predictions first
        try:
            small_df = df[df.avg_cluster == cluster]
            scores = small_df.groupby("frame_nos")['score'].mean()
            max_score_frame_nos = scores.index[np.where(scores == max(scores))[0]]
            fnos.append(max_score_frame_nos)
        except Exception as e:
            #             Case where either the predictions are low or only 1/0 clusters exist
            print(e)
            pass

    return fnos


def hash_the_row(cat, score, bboxs):
    '''
    (category	score	bbox1	bbox2	bbox3	bbox4)
    Output -> 8*6=48 bits:
        - 0-80  (CATERGORY)
        - 2 non-zero after .(left) (SCORE) - 7 bit
        - 2 non-zero after .(left) (BBOX1) - 7 bit
        - 2 non-zero after .(left) (BBOX2) - 7 bitc
        - 2 non-zero after .(left) (BBOX3) - 7 bit
        - 2 non-zero after .(left) (BBOX4) - 7 bit
        -                                  - 35 bits + CATEGORY
    '''
    cat = bin(cat).zfill(9)
    score = bin(
        int("".join([i for i in str(score).split(".")[-1] if i != '0'][:2])))[2:].zfill(8)
    bboxs = "".join([bfloat(bbox) for bbox in bboxs])
    return (
        cat+score+bboxs
    )


'''

from glob import glob
from tqdm import tqdm
model = load_model()
dones = [i.replace("-frame_vectors.pkl", "") for i in glob("*frame_vectors.pkl")]
with tqdm(total=len(glob("*.mp4"))) as progress:
    for url in glob("*.mp4"):
        if url[:-4] not in dones:
            progress.set_description(f"PROCESSING {url}")
            video = Image2Binary(video_path=url, model=model, save_directory="videos")
            video.create_annotations()
            video.get_fnos()
            progress.update(1)
'''
