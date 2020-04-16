import json
import os
import ngtpy
import random

from glob import glob
from tqdm import tqdm
from pathlib import Path


# Building the dataset
# data_path = Path(
#     os.path.join(
#         os.path.expanduser("~"),
#         "friends",
#         "videos",
#     )
# )
data_path = Path(".")

model = load_model()
category_index = pickle.load(open(os.path.join
            (os.path.expanduser("~"), "youtube", "category_index.pkl"), "rb"
            )
        )

query_clip = random.choice([_ath for _ath in glob("*.mp4")])
engine.search_clip(query_clip, model)
print(query_clip)
for temp in engine.temp:
    temp[0].replace("-frame_vectors.pkl", "").replace(".mp4", "")
    print(temp)



'''
Testing The accuracy
'''
positives = 0
total = len(glob("*.mp4"))

for path in glob("*.mp4"):
    query_clip = path
    engine.search_single(query_clip, model)
    for temp in engine.temp[-5:]:
        if temp[0].replace("-frame_vectors.pkl", "").replace(".mp4", "") == query_clip.replace(".mp4", ""):
            positives+=1
        # print(temp)

print(f"Top 5 accuracy is {positives/total}")

