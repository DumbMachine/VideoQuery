import os
import pickle


def convert_annotations_to_df(path):
    '''
    Path to pickle file of the annotations
    '''
    if not os.path.isfile:
        raise FileNotFoundError

    annotations = pickle.load(open(path, "rb"))


    raise NotImplementedError


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
    cat   =   bin(cat).zfill(9)
    score =    bin(int("".join([i for i in str(score).split(".")[-1] if i!='0'][:2])))[2:].zfill(8)
    bboxs =   "".join([bfloat(bbox) for bbox in bboxs])
    return (
        cat+score+bboxs
    )

def frames_to_binary(annotations, frame_array, category_index):
    bins = []
    information = []
    for frame in annotations:
        _information = []
        for cat, bbox, score in zip(
            frame['annotations']['detection_classes'],
            frame['annotations']['detection_boxes'],
            frame['annotations']['detection_scores']
        ):
                if score > 0.6:

                    something = [get_index_from_category(cat, category_index), score]

                    for box in bbox:
                        something.append(box)

                    _information.append(something)
        information.append(_information)

    for fno in frame_array:
        bins.append(
            band_array([hash_the_row(
                elem[0], elem[1], elem[2:]
            ) for elem in information[fno]])
        )

    return bins

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
    for i,j in zip(s1, s2):
        ret += str(1) if i==j else str(0)
    return ret

def bfloat(ffloat):
    if int(ffloat) != ffloat:
        return bin(int("".join([i for i in str(ffloat).split(".")[-1] if i!='0'][:2])))[2:].zfill(8)
    else:
        return bin(int(ffloat)).zfill(8)