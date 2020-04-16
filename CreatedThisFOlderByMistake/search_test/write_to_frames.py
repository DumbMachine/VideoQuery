import pickle
import cv2
import imageio
import os

from glob import glob

videos = sorted(glob("*/*.mp4"), key=lambda x: x.split("/")[0])
idxs = sorted(glob("*/key_frame_indexes.pkl"), key=lambda x: x.split("/")[0])

for video, idx in zip(videos, idxs):
    fnos = pickle.load(open(idx, "rb"))
    video_reel = cv2.VideoCapture(video)
    frames = []
    suc, frame = video_reel.read()
    while suc:
        frames.append(frame)
        suc, frame = video_reel.read()

    imageio.mimsave(
        video.replace("video", "smaller_video"),
        [frames[i] for i in range(len(frames)) if i in fnos]
    )