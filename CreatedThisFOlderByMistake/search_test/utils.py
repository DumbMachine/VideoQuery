from tqdm import tqdm
import pathlib
import os
import cv2
import tensorflow as tf
import numpy as np
import pickle

category_index = pickle.load(open(os.path.join(os.path.expanduser("~"), "youtube", "category_index.pkl"), "rb"))


def get_single_inference(image, model, category_index):
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
