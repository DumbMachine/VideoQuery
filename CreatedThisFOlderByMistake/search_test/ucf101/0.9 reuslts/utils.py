# Import order matters, wrong order causes invalid poitenr error (cv2)
import os
import pathlib
import pickle
import random
from collections import Counter
from glob import glob

import cv2
import imageio
import matplotlib.pyplot as plt
import ngtpy
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.cluster import KMeans
from tqdm import tqdm

from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as vis_util


def read_frame_from_video(video_path, fidx):
    '''
    Read a particular frame from the video
    '''
    video = cv2.VideoCapture(video_path)
    video.set(1, fidx)
    suc, frame = video.read()
    if suc:
        return frame
    return None

def calculate_hist(image, dimension, grayscale=True):
    '''
    Will calculate the histogram of the image in the shape of (dimension, )
    '''
    if grayscale:
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [dimension], [0, 256])

    else:
        hist = cv2.calcHist([image], [0], None, [dimension], [0, 256])

    return hist


def get_batch_inference(frames, model, category_index, batch_size=256):
    '''
    '''
    batch_size = 256
    annotations = {
        "frames": []
    }
    predictions = []
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
    vec = save_key_vectors_from_annotations_with_return(
        annotations,
        category_index
    )

    return vec


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
                "detection_classes" :  [],
                "detection_scores"  :  [],
                "detection_boxes"   :  []
        }


def search(biglist, smallist):
    for i in range(len(biglist)):
        if smallist in biglist[i]:
            return i, biglist[i].index(smallist)
    return None, None


def save_key_vectors_from_annotations_with_return(
    annotation,
    category_index,
    fps=25,
    ):
    '''
    Converting the annotations and saving things
    '''
    start = 0

    vecs = []
    fidxs = []

    for end in range(fps, len(annotation['frames']), fps):
        try:
            batch = annotation['frames'][start : end]
            fno, frame_vecs = get_key_from_batch_with_frame_nos(batch, start, category_index)
            if fno and frame_vecs:
                for _fno, _vec in zip(fno, frame_vecs):
                    fidxs.append(_fno)
                    vecs.append(_vec)

        except Exception as e:
            print(f"{start}-{end}-{e}")
        start = end

    return np.asarray(vecs).flatten()


def get_key_from_batch_with_frame_nos(batch, start, category_index):
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

                    something = [get_index_from_category(cat, category_index), score]

                    for box in bbox:
                        something.append(box)

                    _information.append(something)
        information.append(_information)

    information = [temp for temp in information if temp]
    if not information:
        return None, None
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
            kmeans = KMeans(n_clusters=FRAMES_THRESHOLD).fit(clustering_data[i])
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

    frame_vecs = []
    fnos = []
#     for cluster in range(FRAMES_THRESHOLD):
#         try:
#             frame_vecs.append(
# #             df[df.avg_cluster == cluster].sort_values("score").frame_nos.values[-1]
#                 # df[df.avg_cluster == cluster].sort_values("score").values.flatten()
#                 df[df.frame_nos == df[df.avg_cluster == cluster].sort_values("score").frame_nos.values[0]].loc[:, "category": "bbox4"].values.flatten()
#             )
#             fnos.append(
#                 df[df.avg_cluster == cluster].sort_values("score").frame_nos.values[0]
#             )
#         except:
# #             Case where either the predictions are low or only 1/0 clusters exist
#             pass

    for fno in df.frame_nos:
        frame_vecs.append(df[df.frame_nos == fno].loc[:, "category": "bbox4"].values.flatten())
        fnos.append(fno)
        # df[df.frame_nos == df[df.avg_cluster == cluster].sort_values("score").frame_nos.values[0]].loc[:, "category": "bbox4"].values.flatten()

    return fnos, frame_vecs


def get_single_inference(image, model, category_index=None):
    '''
    Used to get single inferences, mostly used when testing querying
    '''

    if category_index is None:
        category_index = pickle.load(open(os.path.join(os.path.expanduser("~"), "youtube", "category_index.pkl"), "rb"))


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


def visualize_search(frame, name, category_index):
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
