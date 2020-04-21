import os
import cv2
import pickle
import pathlib

import numpy as np
import tensorflow as tf

from utils import fn, CATEGORY_INDEX
from tqdm import tqdm

class FrameLoader:
    """
    Class to load the video frames and apply the appropriate processing to obtain the vectors from each frame
    """

    def __init__(
            self,
            path=None,
            model=None,
            batch_size=256,
            directory="processed",
            verbose=1
    ):
        """
        Will instantiate the FrameLoader object for a single video file
        path: the path of the file to be load as video
        """
        self.path = path
        self.save_timeout = 5 # compress the output every 5 iterations
        self.directory = directory
        self.batch_size = batch_size
        self.current_frame = 0
        self.progress = None
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
                self.progress = tqdm(total=int(total/self.batch_size))

    def _read_video(self):
        """
        Given the path, reads the video and returns the frames from the video
        """
        frames = []
        if self.path:
            video_reel = cv2.VideoCapture(self.path)
        else:
            raise Exception(
                "There was an error with the video path: ", self.path)
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
                    self.progress.set_description(
                        f"[FrameReader] reading frame number: {curr_frame_no}")
                    curr_frame_no += 1
        else:
            while succ:
                frames.append(frame)
                succ, frame = video_reel.read()

        return frames

    def compress_remove_files(self):
        """
        Will compress the files in the self.directory and remove the things from that folder
        """
        #  Check if the there is folder already, if it is then compress the files and add to that
        filename = self.path.split("/")[-1]
        tarpath = os.path.join(self.directory)
        if os.path.isfile(f"processed-{filename}.tar"):
            # append the things in the tarpath
            os.system(f"tar -uvf processed-{filename}.tar {tarpath}")
            # clear the things in that folder to save space
            print("cleaning the files now")
            os.system(f"rm -rf {self.directory}/*")
        else:
            # create the tarpath
            os.system(f"tar -cvf processed-{filename}.tar {tarpath}")


    def load_model(self):
        """
        Loading a simple Tensorflow ObjectDetction Api model if the user doesn't supply a model
        """
        model_name = "ssd_mobilenet_v1_coco_2018_01_28"
        path = os.path.expanduser("~")
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

    def _load_video(self):
        if self.path:
            self.video_reel = cv2.VideoCapture(self.path)
        else:
            raise Exception(
                "There was an error with the video path: ", self.path)

        self.fps = int(self.video_reel.get(cv2.CAP_PROP_FPS))
        self.fnos = int(self.video_reel.get(cv2.CAP_PROP_FRAME_COUNT))
        self._create_progress(total=self.fnos)

    def stream_annotations(self):
        """Stream the video as a generator and then retrieve the annotations from it

        Arguments:
            batch_size {int} -- The amount of frames to be processed at once
        """
        if self.progress is None:
            self._load_video()
        start = 0
        save_timeout = 0
        for end in range(self.batch_size, self.fnos, self.batch_size):
            # getting the batch of images
            frames = self._read_video_in_batches(self.video_reel)
            self.build_annotations(frames)
            print("this is some random text", save_timeout)
            save_timeout += 1
            if save_timeout == self.save_timeout:
                save_timeout = 0
                self.compress_remove_files()

    def _read_video_in_batches(self, video_reel):
        """Read the batch frames from the video reel`

        Arguments:
            video_reel {cv2.VideoCaputre} --The video real from which the video is to be read
        """
        frames = []
        for _ in range(self.batch_size):
            self.progress.set_description(
                f"[Reading Video] frame number: {self.current_frame + _}")
            success, frame = video_reel.read()
            if not success:
                raise Exception("All the frames have finished")
            frames.append(frame)
        self.current_frame += _
        return frames

    # TODO: Build a video streamer (generator)

    def build_annotations(self, frames):
        """Build the pkl file which has the object appearance information from each frame
        """
        predictions = []
        input_tensor = tf.convert_to_tensor(
            np.asarray(frames)
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
            # self.progress.set_description(
            #     f"Running with the local counter {local_counter}")
            # local_counter += 1

        self.progress.update(1)

        path = os.path.join(self.directory, self.path.split("/")[-1])

        path = path + \
            f"{self.current_frame}:{self.current_frame-self.batch_size}"

        pickle.dump(
            self.annotations,
            open(
                f"{path}-annotations.pkl", "wb"
            )
        )