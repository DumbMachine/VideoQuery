import os
import pathlib
import pickle
import random
from glob import glob

import cv2
import imageio
import matplotlib.pyplot as plt
import ngtpy
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as vis_util

dims = 12
if dims:
    frame_idxs = sorted(glob("*/key_frame_indexes.pkl"), key=lambda x: x.split("/")[0])[:]
    frame_reps = sorted(glob(f"*/key_frame_reps-{dims}.pkl"), key=lambda x: x.split("/")[0])[:]
else:
    frame_idxs = sorted(glob("*/key_frame_indexes.pkl"), key=lambda x: x.split("/")[0])[:]
    frame_reps = sorted(glob("*/key_frame_reps.pkl"), key=lambda x: x.split("/")[0])[:]

category_index = pickle.load(open(os.path.join(os.path.expanduser("~"), "youtube", "category_index.pkl"), "rb"))

'''
# TODO: Make a function to benchmark
# Add progress status for building the indexes
'''
class SearchEngine():
    '''
    Class Object For building the Search Engine and Searching

    [USAGE]:
    engine = SearchEngine(idxs_path=frame_idxs, reps_path=frame_reps, dimension=6)
    engine._build_index()
    engine._build_dataset()
    '''
    def __init__(
        self,
        idxs_path,
        reps_path,
        progress=True,
        dimension_bins=32):
        '''
        Initializing the SearchEngine Object
        @parameters:
        ------------
        - idxs_path: The frame index array, which has the frame index of each frame in the Original Video
        - reps_path: The frame representation array, which has the frame representations of each frame
        '''

        self.dims = 6
        self.max_dims = 20  # FIXME: Dont hard code this
        self.dims_bin = dimension_bins
        # self.progress = progress

        self.idxs_path = idxs_path
        self.reps_path = reps_path

        self.data = {}
        self.index = {}
        self.mapping = {}


    # def _dimensions(self):
    #     '''
    #     Get the dimension of each vector (<nos-of-detections>*6 + bins_dimension
    #     '''

    def _build_index(self):
        '''
        Will initialize the indexes and data maps
        '''
        for i in range(self.max_dims):
            tempfile = str.encode(f"{(i+1) + self.dims_bin}-Index-tmp")
            ngtpy.create(tempfile, self.dims*(i+1) + self.dims_bin)
            self.index[self.dims*(i+1) + self.dims_bin] = ngtpy.Index(tempfile)

            self.data[self.dims*(i+1) + self.dims_bin] = []


    def _build_dataset(self, progress=True):
        '''
        Will read the pickle files and load the corresponding frame representation and index in the appropriate places
        '''
        if progress:
            with tqdm(total=len(self.reps_path)) as progress:
                for vecs_path, frame_idxs_path in zip(self.reps_path, self.idxs_path):
                    vecs = pickle.load(open(vecs_path, "rb"))
                    frame_idxs = pickle.load(open(frame_idxs_path, "rb"))
                    vpath = vecs_path.replace("/key_frame_reps.pkl", ".avi")

                    self.vecs = vecs
                    self.frame_idxs = frame_idxs
                    if len(vecs) and len(frame_idxs):
                        self.mapping[vecs_path] = []
                        for vec, idx in zip(vecs, frame_idxs):
                            # TODO: Check for error here, if None is returned from hist
                            # hist = self._add_histogram(self.dims_bin, idx, vpath).reshape(self.dims_bin)
                            # vec = np.append(vec, hist)
                            self.data[len(vec)].append(vec)
                            self.mapping[vecs_path].append(
                                f"{len(vec)}-{len(self.data[len(vec)])}-{idx}"
                            )
                        progress.update(1)


        else:
            raise NotImplementedError

    def _insert_dataset(self):
        '''
        Will insert the frame representations in the appropriate index
        '''
        for key in self.data.keys():
            if self.data[key]:
                self.index[key].batch_insert(self.data[key])

    def _add_histogram(self, dimension, frame_idx, vpath, gray=True):
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


    def search_frame(
        self,
        frame,
        model,
        category_index,
        predictions=1,
        source=False,
        visualize=False,
        video_visualize=False,
        gray=True):
        '''
        Will search for the particular frame in the indexes

        @parameters:
        ------------
        - frame: the image array
        - category_index: the category_index used for mapping
        - predictions: the number of returned frames
        - source: flag for returning the source path of the image
        '''
        qvec = np.asarray(
            get_single_inference(frame, model, category_index)
        ).flatten()
        if not len(qvec):
            return "No Objects in the image, thus cannot search", None

        if gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([frame], [0], None, [self.dims_bin], [0, 256]).reshape(self.dims_bin)
        qvec = np.append(qvec, hist)

        predictions = []
        image_array = []

        result = self.index[len(qvec)].search(qvec, 3)

        for idx, distance in result:
            for key in self.mapping.keys():
                for item in self.mapping[key]:
                    if f"{len(qvec)}-{idx}" == "-".join(item.split("-")[:-1]):
                        predictions.append(key)
                        if self.dims_bin:
                            image_array.append(
                                (item.split("-")[-1], key.replace(f"/key_frame_reps-{self.dims_bin}.pkl", ".avi"))
                            )
        output = {}
        self.image_array = image_array
        for pred in predictions:
            if pred in output.keys():
                output[pred] += 1
            else:
                output[pred] = 1

        if video_visualize or visualize:
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
                #TODO: Change names of the query stored
                plt.savefig(f"figures/query.jpeg")
                video_frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)

                if video_visualize:
                    video_frame = video_frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    plt.clf()
                    return output, video_frame
        return output

    def search_clip(
        self,
        clip,
        model,
        category_index,
        batch_size=256,
        neighbours=3
    ):
        '''
        Batch search the whole clip
        '''
        annotations = {
            "frames": []
        }
        if len(frames) < batch_size:
            input_tensor = tf.convert_to_tensor(
                np.asarray(frames)
            )
            prediction = model(input_tensor)
            for idx in range(
                len(prediction['detection_scores'])
            ):
                output_dict = {
                    "detection_classes": prediction["detection_classes"][idx],
                    "detection_scores" : prediction["detection_scores"][idx],
                    "detection_boxes"  : prediction["detection_boxes"][idx].numpy()
                }
                annotations['frames'].append(
                    {
                        "annotations": fn(output_dict, category_index)
                    }
                )

        else:
            start = 0
            for end in range(batch_size, len(frames), batch_size):
                input_tensor = tf.convert_to_tensor(
                    np.asarray(frames[ start : end ])
                )
                prediction = model(input_tensor)
                for idx in range(
                    len(prediction['detection_scores'])
                ):
                    output_dict = {
                        "detection_classes": prediction["detection_classes"][idx],
                        "detection_scores" : prediction["detection_scores"][idx],
                        "detection_boxes"  : prediction["detection_boxes"][idx].numpy()
                    }
                    annotations['frames'].append(
                        {
                            "annotations": fn(output_dict, category_index)
                        }
                    )

                start = end

        raise NotImplemented


    # TODO: Pred all the images all at once
    def search_clip_full(self, clip, model, category_index, predictions=1, visualize=False):
        '''
        Given the video clip, this function will return the total average prediction
        '''
        prediction = []
        video = []
        for frame in clip:
            if visualize:
                pred, video_frame = self.search_frame(frame, model, category_index, predictions, video_visualize=True)
                if pred is not None and video_frame is not None:
                    video.append(video_frame)
            else:
                pred = self.search_frame(frame, model, category_index, predictions)
            if type(pred) != str:
                prediction.append(pred)
        
        self.prediction = prediction

        temp = []
        for pred in prediction:
            for item in pred.items():
                temp.append(item)
        # Counting frequency in temp
        freq = {}
        for pred in temp:
            if pred[0] in freq.keys():
                freq[pred[0]] += pred[1]
            else:
                freq[pred[0]] = pred[1]

        freq = {k: v for k, v in sorted(freq.items(), key=lambda item: item[1], reverse=True)}

        if visualize:
            if video:
                imageio.mimsave("figures/query.mp4", [frame for frame in video if len(frame)])


        return list(freq.items())[:5]

engine = SearchEngine(idxs_path=frame_idxs, reps_path=frame_reps, dimension_bins=dims)
engine._build_index()
engine._build_dataset()
engine._insert_dataset()

model = load_model()


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


positives = 0
total = 0
iterations = 100
for i in range(iterations):
    print(i)
    query_video = random.choice(frame_reps).replace(f"/key_frame_reps-{dims}.pkl", ".avi")
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
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            hist = cv2.calcHist([frame], [0], None, [dims], [0, 256])
            qvec = np.append(
                qvec, hist
            )
            result = engine.index[len(qvec)].search(qvec, 3)

            for idx, distance in result:
                for key in engine. mapping.keys():
                    for item in engine. mapping[key]:
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

positives/float(i)


# query_video = random.choice(frame_reps).replace(f"/key_frame_reps-{dims}.pkl", ".avi")
# frames = read_video(query_video)
# image = random.choice(frames)
# (query_video, engine.search_frame(image, model, category_index, visualize=True))

# query_start_indx = random.randint(0, int(len(frames)*0.5))
# query_clip = frames[query_start_indx: int(query_start_indx+len(frames)*0.2)]

# engine.search_clip_full(frames, model, category_index, visualize=True)


# # qvec = np.asarray(
# #             get_single_inference(image, model, category_index)
# #         ).flatten()

# # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# # hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).reshape(32)
# # qvec = np.append(qvec, hist)
# # engine.index[len(qvec)].search(qvec, 3)
# # # query_start_indx = random.randint(0, int(len(frames)*0.5))
# # # clip = frames[query_start_indx: int(query_start_indx+len(frames)*0.2)]


# # # engine.search_clip_full(clip, model, category_index)


# #         predictions = []

# #         image_array = []



# #         result = engine.index[len(qvec)].search(qvec, 3)

# #         for idx, distance in result:
# #             for key in engine.mapping.keys():
# #                 for item in engine.mapping[key]:
# #                     if f"{len(qvec)}-{idx}" == "-".join(item.split("-")[:-1]):
# #                         predictions.append(key)
# #                         image_array.append(
# #                             (item.split("-")[-1], key.replace("/key_frame_reps.pkl", ".avi"))
# #                         )
# #         output = {}
# #         for pred in predictions:
# #             if pred in output.keys():
# #                 output[pred] += 1
# #             else:
# #                 output[pred] = 1
