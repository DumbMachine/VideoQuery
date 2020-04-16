import os
from glob import glob
from tqdm import tqdm
import pickle

from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from collections import Counter

'''
Converting the annotations to frame_vector representations and also storing the frame_number of each vector representations
'''
annotations = glob("*annotations.pkl")
category_index = pickle.load(open(os.path.join
            (os.path.expanduser("~"), "youtube", "category_index.pkl"), "rb"
            )
        )

def save_key_vectors_from_annotations(
    path,
    fps=25
    ):
    '''
    Converting the annotations and saving things
    '''
    start = 0

    vecs = []
    fidxs = []
    annotation = pickle.load(open(path, "rb"))

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

    pickle.dump(
        vecs,
        open(f"{path}-VECS.pkl", "wb")
    )

    pickle.dump(
        fidxs,
        open(f"{path}-FRAME-INDEXES.pkl", "wb")
    )


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
    for cluster in range(FRAMES_THRESHOLD):
        try:
            frame_vecs.append(
#             df[df.avg_cluster == cluster].sort_values("score").frame_nos.values[-1]
                # df[df.avg_cluster == cluster].sort_values("score").values.flatten()
                df[df.frame_nos == df[df.avg_cluster == cluster].sort_values("score").frame_nos.values[0]].loc[:, "category": "bbox4"].values.flatten()
            )
            fnos.append(
                df[df.avg_cluster == cluster].sort_values("score").frame_nos.values[0]
            )
        except:
#             Case where either the predictions are low or only 1/0 clusters exist
            pass

    return fnos, frame_vecs


def get_index_from_category(category, category_index):
    '''
    Function to return the inverse mapping from category_index (the dict with all the category names)
    '''
    for key in category_index.keys():
        if category_index[key]['name'] == category:
            return key

def search(biglist, smallist):
    for i in range(len(biglist)):
        if smallist in biglist[i]:
            return i, biglist[i].index(smallist)
    return None, None


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



# Taking 5 items from annotations and making their 
for ano_path in annotations[:5]:
    save_key_vectors_from_annotations(ano_path)

'''
testing if the above worked
'''

vecs = [
    pickle.load(open(i, "rb")) for i in glob("*-VECS.pkl") if pickle.load(open(i, "rb"))
]
frame_idx = [
    pickle.load(open(i, "rb")) for i in glob("*-FRAME-INDEXES.pkl") if pickle.load(open(i, "rb"))
]
