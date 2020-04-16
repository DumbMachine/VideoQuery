'''
Rewrite of pipeline, for smarter and better things overall, major things fixed are:
- Reading video in chunks, for larget videos
    -
'''



import os
import cv2
import pafy
import pickle

from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as vis_util


# TODO:
    # - Function to Write to disk the thing

class Video2Vector:
    '''
    USAGE:
    ------

    1. Loading video via URL:
    video = Video2Vector(
        url="https://www.youtube.com/watch?v=erAQ9LkftwA"
    )
    video._download_video()

    '''
    def __init__(
        self,
        url=None,
        video_path=None,
        category_index=None,
        save_directory=os.path.curdir
    ):
        '''
        @parameters:
        ---------------
        - url: The url of the youtube video, if you wish to download the video
        - video_path: The path to the video file, you wish to use
        - save_directory: [CWD: default] Directory where you want to save the files(pkl and downloaded video(if youtube is used))

        '''
        if url is None and video_path is None:
            raise Exception("Provide one source for video")

        self.url = url
        self.video_path = video_path
        self.save_directory = save_directory
        self.category_index = category_index

        # Creating the save_directory:
        if not os.path.isdir(self.save_directory):
            os.makedirs(self.save_directory, exist_ok=True)

    def download_video_youtube(
        self,
        stream="normal:mp4@640x360w",
        quiet=True):
        '''
        Function download the video off of youtube

        @parameters:
        ------------
        quiet: Progress bar display flag

        '''
        # List of Supported Streams
        STREAMS = [
            "normal:webm@640x360",
            "normal:mp4@640x360",
            "normal:flv@320x240",
            "normal:3gp@320x240",
            "normal:3gp@176x144",
            "video:m4v@854x480",
            "video:m4v@640x360",
            "video:m4v@426x240",
            "video:m4v@256x144",
        ]
        stream = "normal:mp4@640x360"

        video = pafy.new(self.url)
        dl_stream = None

        for _stream in video.streams:
            if stream == str(_stream):
                dl_stream = _stream
                break

        if dl_stream is None:
            raise Exception("The required quality is not available for this video")

        #  Sometimes titles changes automatically

        self.video_path = os.path.join(
            self.save_directory, "video.mp4"
        )

        dl_stream.download(self.video_path, quiet=quiet)

    def read_video(self, progress=False):
        '''
        Given the path, reads the video and returns the frames from the video
        '''
        frames = []
        video_reel = cv2.VideoCapture(self.video_path)

        self.fnos = int(video_reel.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(video_reel.get(cv2.CAP_PROP_FPS))
        succ , frame = video_reel.read()
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


    def read_video_blocks(
        self,
        progress=False,
        multiplier=10
        ):
        '''
        Returns an iterator to the video in iterations of <block_size>
        @paramters:
        ------------
        progress: flag to show the progres
        block_size: The size with which to read the video [None if you want to get all the frames]

        '''
        assert type(multiplier) == int, "multiplier should be a integer"
        block_size = self.fps * multiplier
        video_reel = cv2.VideoCapture(self.video_path)
        self.fps = int(video_reel.get(cv2.CAP_PROP_FPS))
        self.fnos = int(video_reel.get(cv2.CAP_PROP_FRAME_COUNT))

        # FIXME: hardcoded the block_size to frame size, so frame clustering order isn't disturbed
        if block_size:
            start = 0
            for end in range(block_size, self.fnos, block_size):
                video_reel.set(1, start)
                _frames = []
                suc, frame = video_reel.read()
                while suc and start < end:
                    _frames.append(frame)
                    suc, frame = video_reel.read()
                    start+=1

                yield _frames

            video_reel.set(1, start)
            _frames = []
            suc, frame = video_reel.read()
            while suc and start < self.fnos:
                _frames.append(frame)
                suc, frame = video_reel.read()
                start+=1

            yield _frames


    # FIXME:
        # Fix Batch Mode
    def predict(
        self,
        model,
        read_video="not_batch",
        multiplier=10,
        progress=False
    ):
        '''

        '''


        self.annotations = {
            "frames": []
        }

        if read_video == "batch":
            batch_size= self.fps * multiplier
            for frames in self.read_video_blocks(multiplier=multiplier):
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
                        self.annotations['frames'].append(
                            {
                                "annotations": fn(output_dict, self.category_index)
                            }
                        )
        else:
            start = 0
            predictions = []
            frames = self.read_video()
            batch_size= self.fps * multiplier
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
                    self.annotations['frames'].append(
                        {
                            "annotations": fn(output_dict, self.category_index)
                        }
                    )

                start = end

        pickle.dump(
            self.annotations,
            open(
                f"{self.save_directory}/annotations.pkl", "wb"
            )
        )


    def key_frames(self):
        save_key_vectors_from_annotations(self.save_directory)





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
    annotation = pickle.load(open(f"{path}/annotations.pkl", "rb"))

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
        open(f"{path}/key_frame_reps.pkl", "wb")
    )

    pickle.dump(
        fidxs,
        open(f"{path}/key_frame_indexes.pkl", "wb")
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


def format_title(string):
    items = []
    for item in string.split(" "):
        items.append(re.sub('[^a-zA-Z0-9-_*.]', '', item))
    return " ".join(items)
