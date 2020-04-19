import os
import cv2
import pickle
import random
import imageio
import pathlib
import operator
import functools

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from glob import glob
from collections import Counter
from sklearn.cluster import KMeans
from joblib import Parallel, delayed


# Defining the important paths
REPO_PATH = os.curdir
DATA_PATH = os.path.join(REPO_PATH, ".data")
SAMPLE_VIDEO = os.path.join(DATA_PATH, "sample_sample.mp4")
CATEGORY_INDEX = os.path.join(DATA_PATH, "category_index.pkl")

FRAME_VIDEO_PATH = os.path.join(DATA_PATH, "videos")
FRAME_NUMBER_PATH = os.path.join(DATA_PATH, "frame_numbers")
FRAME_VECTORS_PATH = os.path.join(DATA_PATH, "vectors")
FRAME_ANNOTATION_PATH = os.path.join(DATA_PATH, "annotations")

category_index = pickle.load(open(CATEGORY_INDEX, "rb"))


def index_dataset_from_folder(folder):
    files = glob(f"{folder}/**")
    for file in files:
        key_frames_parallel(file)


# In the iterator, yield the BATCH and not the whole annotations
def key_frames_parallel(path, n_jobs=-1, FPS=30):
    """
    Parallelize the clustering using Joblib
    """

    filename = path.split("/")[-1].split(".")[0]
    # checking if the files already exist
    patha = [os.path.join(FRAME_VECTORS_PATH, filename+".pkl-failed"),
             os.path.join(FRAME_NUMBER_PATH, filename+".pkl-failed")]
    pathb = [os.path.join(FRAME_VECTORS_PATH, filename+".pkl"),
             os.path.join(FRAME_NUMBER_PATH, filename+".pkl")]
    patha = [os.path.isfile(file) for file in patha]
    pathb = [os.path.isfile(file) for file in pathb]

    if patha == [True, True] or pathb == [True, True]:
        print("skipping the files")
        return

    annotations = pickle.load(open(path, "rb"))
    category_index = pickle.load(open(CATEGORY_INDEX, "rb"))
    arg_instances = [(start, annotations['frames'][start: start+FPS])
                     for start in range(0, len(annotations['frames']), FPS)]

    results = Parallel(n_jobs=-1, verbose=0, backend="loky")(
        map(delayed(get_key_from_batch), arg_instances))

    vecs = []
    fidxs = []
    if results is None:
        pickle.dump(
            vecs,
            open(os.path.join(FRAME_VECTORS_PATH, filename+".pkl-failed"), "wb")
        )
        pickle.dump(
            fidxs,
            open(os.path.join(FRAME_NUMBER_PATH, filename+".pkl-failed"), "wb")
        )

    else:
        for idx, vec in results:
            if idx is not None and vec is not None:
                vecs.extend(vec)
                fidxs.extend(idx)

        pickle.dump(
            vecs,
            open(os.path.join(FRAME_VECTORS_PATH, filename+".pkl"), "wb")
        )
        pickle.dump(
            fidxs,
            open(os.path.join(FRAME_NUMBER_PATH, filename+".pkl"), "wb")
        )
        # print("The filenames used by the parallel_operation ",
        #     filename, FRAME_VECTORS_PATH, filename+".pkl", FRAME_NUMBER_PATH, filename+".pkl")


def get_key_from_batch(something):
    '''
    Returns the key frames from the given batch of frames
    '''
    start, batch = something
    threshold = 5
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
        print("this file was wrong")
        return [], []
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
    clustering_data_prediction = []
    for i in range(len(clustering_data)):
        if len(clustering_data[i]) >= threshold:
            kmeans = KMeans(n_clusters=threshold).fit(
                clustering_data[i])
            clustering_data_prediction.append(kmeans.labels_)

    indx = len(clustering_data)
    for i in range(len(clustering_data)):
        if len(clustering_data[i]) < threshold:
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
    frame_vecs = []
    for cluster in range(threshold):
        # Taking the frame with the most number of predictions first
        try:
            small_df = df[df.avg_cluster == cluster]
            scores = small_df.groupby("frame_nos")['score'].mean()
            max_score_frame_nos = scores.index[np.where(
                scores == max(scores))[0]][0]
            fnos.append(max_score_frame_nos)
            frame_vecs.append(small_df[
                small_df.frame_nos == max_score_frame_nos
            ].loc[:, "category":"bbox4"].values.flatten()
            )

        except Exception as e:
            #             Case where either the predictions are low or only 1/0 clusters exist
            print(e)
            pass

    # sorting the vectors according to the index of their respective frames
    whoosh = zip(fnos, frame_vecs)
    whoosh = sorted(whoosh, key=lambda x: x[0])
    fnos = []
    frame_vecs = []
    for fno, vec in whoosh:
        fnos.append(fno)
        frame_vecs.append(vec)
    return fnos, frame_vecs


def save_vec_with_histograms(filename):
    """Save the frame vectors with the corressponding histogram

    Arguments:
        obj {list} -- list of a few values required
    """
    dims = 18
    vpath = os.path.join(FRAME_VIDEO_PATH + f"/{filename}")
    vecs_path = os.path.join(FRAME_VECTORS_PATH +
                             f"/{filename.replace('.avi', '.pkl')}")
    frame_idxs_path = os.path.join(
        FRAME_NUMBER_PATH + f"/{filename.replace('.avi', '.pkl')}")

    vecs = pickle.load(open(vecs_path, "rb"))
    frame_idxs = pickle.load(open(frame_idxs_path, "rb"))
    savename = filename.replace(".avi", ".pkl")
    directory = os.path.join(DATA_PATH, f"vectors-{dims}")

    if os.path.isfile(os.path.join(directory, savename)):
        return

    vecs_hist = []
    if vecs and frame_idxs:
        for vec, idx in zip(vecs, frame_idxs):
            # TODO: Check for error here, if None is returned from hist
            hist = get_hist(dims, idx, vpath).reshape(dims)
            vec = np.append(vec, hist)
            vecs_hist.append(
                vec
            )

        if not os.path.isdir(directory):
            os.makedirs(directory)

        pickle.dump(
            vecs_hist, open(os.path.join(directory, savename), "wb")
        )


def get_hist(dimension, frame_idx, vpath, gray=True):
    """Util function when we wanna add in the histogram

    Arguments:
        dimension {int} -- bins for the histogram
        frame_idx {int} -- frame_idx in the video
        vpath {string} -- Path of the video

    Keyword Arguments:
        gray {bool} -- grayscale  (default: {True})

    Returns:
        [type] -- [description]
    """
    video_reel = cv2.VideoCapture(vpath)
    video_reel.set(1, frame_idx)
    suc, image = video_reel.read()
    if suc:
        if gray:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        hist = cv2.calcHist([image], [0], None, [dimension], [0, 256])
        return hist
    else:
        return []


def fn(output_dict, category_index):
    """[summary]

    Arguments:
        output_dict {[type]} -- [description]
        category_index {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
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


def get_index_from_category(category, category_index):
    '''
    Function to return the inverse mapping from category_index (the dict with all the category names)
    '''
    for key in category_index.keys():
        if category_index[key]['name'] == category or key == category:
            return key


def search(biglist, smallist):
    for i in range(len(biglist)):
        if smallist in biglist[i]:
            return i, biglist[i].index(smallist)
    return None, None

# FIXME: This needs to be fized, get_batch was changed


def compare_segments(batch=[], frames=[], frame_size=3, save=True, filename="test_comparison.mp4"):
    """Will compare the original video with the key frames obtained from the video

    Arguments:
        frames {list} -- [description] (default: {[]})
        frame_size {int} -- The number of frames to be extracted from the batch of images passed as frames (default: {[]})
    """
    # obtaining the key frames for this batch
    category_index = pickle.load(open(CATEGORY_INDEX, "rb"))
    fnos = get_key_from_batch(batch=batch, start=0,
                              category_index=category_index)

    # generating a movie from the clips obtained as the key frames
    key_frames = []
    for _ in range(int(len(frames)/frame_size)):
        for fno in fnos:
            key_frames.append(frames[fno])

    video = []
    for a, b in zip(frames, key_frames):
        image = twoimages_oneplot(a, b)
        video.append(image)

    if save:
        imageio.mimsave(filename, video)
    else:
        return video


def twoimages_oneplot(image, another_image):
    """Display two images side by side

    Arguments:
        image {numpy.array} -- First image
        another_image {numpy.array} -- Second image
    """
    fig = plt.figure()
    fig.tight_layout(pad=0)
    image1 = plt.subplot(121)
    image2 = plt.subplot(122)
    image1.axis('off')
    image2.axis('off')
    _ = image1.imshow(image)
    _ = image2.imshow(another_image)

    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(
        fig.canvas.get_width_height()[::-1] + (3,))
    plt.clf()
    plt.close('all')

    return image_from_plot


def parallel_frames(batch):
    '''
    processing function
    '''
    # start, end, annotation = object
    vecs = []
    fidxs = []
    try:
        # batch = annotation['frames'][start : end]
        fno, frame_vecs = get_key_from_batch_with_frame_nos(
            batch, start, category_index)
        if fno and frame_vecs:
            for _fno, _vec in zip(fno, frame_vecs):
                fidxs.append(_fno)
                vecs.append(_vec)
        return vecs, fidxs
    except Exception as e:
        return [], []
