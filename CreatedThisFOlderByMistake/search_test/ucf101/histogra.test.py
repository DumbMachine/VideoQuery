import cv2
import matplotlib.pyplot as plt

file = "Titanic 3D _ _Third Class Dance_ _ Official Clip HD"
video = cv2.VideoCapture(file)
frames = []
suc, frame = video.read()
while suc:
    frames.append(frame)
    suc, frame = video.read()

len(frames)

# Calculating time required for histograms
%%time
hists = []
for image in frames:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hists.append(hist)

imageio.mimsave("temp.gif", hists)



