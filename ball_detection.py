import os
import cv2
import mmcv
from misc import formal_list


def split_video(video_id):
    os.makedirs(f"ball_output/{video_id:05}", exist_ok=True)
    video = mmcv.VideoReader(f"data/train/{video_id:05}/{video_id:05}.mp4")
    for img_id, img in enumerate(video):
        cv2.imwrite(f"ball_output/{video_id:05}/{img_id:04}.jpg", img)

# for video_id in formal_list:
# split_video(1)

import cv2
import numpy as np
img = cv2.imread("ball_output/00001/0120.jpg", cv2.IMREAD_GRAYSCALE)
# img = np.ones_like(img) * 255 - img
# img = img[160:720-160, 160:1280-160]

params = cv2.SimpleBlobDetector_Params()
params.minThreshold = 160
params.maxThreshold = 220
# params.filterByArea = True
# params.maxArea = 50
# params.maxArea = 100
# params.filterByCircularity = True
# params.minCircularity = 0.2

# # 凸度
# params.filterByConvexity = True
# params.minConvexity = 0.5

# # 慣性比
# params.filterByInertia = True
# params.minInertiaRatio = 0.5
detector = cv2.SimpleBlobDetector_create(params)

keypoints = detector.detect(img)
img_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("Keypoints", img_with_keypoints)
cv2.waitKey(0)