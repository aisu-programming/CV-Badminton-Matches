import cv2
import mmcv
import numpy as np


videoReader = mmcv.VideoReader("data/train/00001/00001.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = videoReader.fps
size = (videoReader.width, videoReader.height)
videoWriter = cv2.VideoWriter("test.mp4", fourcc, fps, size)
video = np.uint8(videoReader[:])

bg = background = cv2.imread("output/background/avg_without_players/00001.png")
bg = np.expand_dims(bg, axis=0)
diff = np.sum((video-bg)**2, axis=-1)
video[diff<300] = [0,0,0]

for frame in video: videoWriter.write(frame)
videoWriter.release()