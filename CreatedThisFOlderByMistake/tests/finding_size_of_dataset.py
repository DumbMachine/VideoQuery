import json
import os
import cv2
from glob import glob
from tqdm import tqdm
from pathlib import Path


# Building the dataset

total_time = 0
total_frames = 0

videos = glob("*.mp4")

with tqdm(total=len(videos)) as progress:
    for video in videos:
        cap = cv2.VideoCapture(str(video))
        fnos = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        duration = float(fnos) / float(fps)

        total_time+=duration
        total_frames+=fnos
        progress.update(1)

total_time/60, total_frames