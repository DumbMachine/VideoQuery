"""
modified script to run from kaggle
"""

import os
import cv2
import pickle
import pathlib

import numpy as np
import tensorflow as tf

from tqdm import tqdm_notebook as tqdm

CATEGORY_INDEX = "/kaggle/input/category-index/category_index.pkl"


import os
import cv2
import pickle
import pathlib

import numpy as np
import tensorflow as tf

from tqdm import tqdm_notebook as tqdm
CATEGORY_INDEX = "/kaggle/input/category-index/category_index.pkl"

class FrameLoader:
    """
    Class to load the video frames and apply the appropriate processing to obtain the vectors from each frame
    """
    def __init__(
            self,
            path=None,
            model=None,
            batch_size=256,
            filename=None,
            verbose=1
    ):
        """
        Will instantiate the FrameLoader object for a single video file
        path: the path of the file to be load as video
        """
        self.path = path
        self.batch_size = batch_size
        self.filename = filename
        if model is None:
            self.model = self.load_model()
        else:
            self.model = model

        self.verbose = verbose
        self.annotations = {
            "frames": []
        }
        # loading the video frames:
        self.category_index = pickle.load(open(CATEGORY_INDEX, 'rb'))

    # TODO: Update the `total` varaible with the total number of steps
    def _create_progress(self, total=None):
        """
        logging the total process of creation of frames and other things like that
        steps in this btich:
        1. reading the video
        2. loading the model
        3. starting the creation process
            - mention the frame number when generating the thing
        """
        if self.verbose == 1:
            # creating the normal thing
            if self.batch_size > self.fnos:
                self.progress = tqdm(total=total)
            else:
                self.progress = tqdm(total=int(total/256))


    def _read_video(self):
        """
        Given the path, reads the video and returns the frames from the video
        """
        frames = []
        if self.path:
            video_reel = cv2.VideoCapture(self.path)
        else:
            raise Exception("There was an error with the video path: ", self.path)
        # else:
        #     video_reel = cv2.VideoCapture(self.path+"/video.mp4")

        self.fnos = int(video_reel.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(video_reel.get(cv2.CAP_PROP_FPS))
        self._create_progress(total=self.fnos)
        succ, frame = video_reel.read()
        curr_frame_no = 0
        if self.verbose == 1:
                while succ:
                    frames.append(frame)
                    succ, frame = video_reel.read()
                    self.progress.set_description(f"[FrameReader] reading frame number: {curr_frame_no}")
                    curr_frame_no += 1
        else:
            while succ:
                frames.append(frame)
                succ, frame = video_reel.read()

        return frames

    def load_model(self):
        """
        Loading a simple Tensorflow ObjectDetction Api model if the user doesn't supply a model
        """
        model_name="ssd_mobilenet_v1_coco_2018_01_28"
        path=os.path.expanduser("~")
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

    def stream_annotations(self):
        """Stream the video as a generator and then retrieve the annotations from it

        Arguments:
            batch_size {int} -- The amount of frames to be processed at once
        """
        if self.path:
            video_reel = cv2.VideoCapture(self.path)
        else:
            raise Exception("There was an error with the video path: ", self.path)

        self.fps = int(video_reel.get(cv2.CAP_PROP_FPS))
        self.fnos = int(video_reel.get(cv2.CAP_PROP_FRAME_COUNT))
        self._create_progress(total=self.fno)
        start = 0
        for end in range(self.batch_size, fnos, self.batch_size):
            # getting the batch of images
            frames = self._read_video_in_batches(self, video_reel)
            self.build_annotations(frames)

    def _read_video_in_batches(self, video_reel):
        """Read the batch frames from the video reel`

        Arguments:
            video_reel {cv2.VideoCaputre} --The video real from which the video is to be read
        """
        frames = []
        for _ in range(self.batch_size):
            frame, success = video_reel.read()
            frames.append(frame)
        return frames


    # TODO: Build a video streamer (generator)
    def build_annotations(self, frames):
        """Build the pkl file which has the object appearance information from each frame
        """

        if len(frames) < self.batch_size:
            input_tensor = tf.convert_to_tensor(np.asarray(frames))
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
                self.progress.update(1)

        else:
            start = 0
            predictions = []

            for end in range(self.batch_size, len(frames), self.batch_size):
                input_tensor = tf.convert_to_tensor(
                    np.asarray(frames[start: end])
                )
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
                self.progress.update(1)

        if self.filename:
            path = self.filename
        else:
            path = self.path

        pickle.dump(
            self.annotations,
            open(
                f"{path}-annotations.pkl", "wb"
            )
        )
