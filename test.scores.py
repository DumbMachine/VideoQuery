import pickle
import random

from glob import glob
from search_wrapper import VideoEngine

size = None
top = 1000

if size:
    vecs_directory=f"/home/dumbmachine/code/SVMWSN/.data/vectors-{size}"
else:
    vecs_directory="/home/dumbmachine/code/SVMWSN/.data/vectors"


engine = VideoEngine(
    fnos_directory="/home/dumbmachine/code/SVMWSN/.data/frame_numbers",
    vecs_directory=vecs_directory
)
engine.build_indexes()
engine.build_dataset()
engine.insert_dataset()


def test_something():
    plus = 0
    query_clips = random.sample([_ath for _ath in glob("/home/dumbmachine/code/SVMWSN/.data/vectors-18/*")], 1000)
    for intr, query_clip in enumerate(query_clips):
        print(intr, len(query_clips))
        vectors = pickle.load(open(query_clip, "rb"))
        if len(vectors):
            engine.search_vectors(query_row=vectors[0])
            predictions = [i[0] for i in engine.temp[:top]]
            if query_clip in predictions:
                plus+=1

    print(plus/len(query_clips))


"""
Information here:
vector: has the normal and naive vector: dimension is {6, +6, ...}
vector-12: has the normal and naive vector: dimension is {12, +6, ...}
vector-12: has the normal and naive vector: dimension is {18, +6, ...}
vector-18: has the normal and naive vector: dimension is {24, +6, ...}
"""