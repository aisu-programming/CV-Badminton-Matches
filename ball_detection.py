import cv2
import numpy as np


img = cv2.imread("ball_output/00001/0113.jpg")
# img = np.ones_like(img) * 255 - img
# img = img[160:720-160, 160:1280-160]
minus_img = np.ones_like(img) * 190
minus_img = np.minimum(img, minus_img)
white_img = np.ones_like(img)
white_img[np.where(np.greater_equal(np.sum((img-minus_img), axis=-1), 10))] = [ 0, 0, 0 ]
white_img[np.where(np.less(np.sum((img-minus_img), axis=-1), 10))] = [ 255, 255, 255]

img = np.uint8(img*(1/3) + white_img*(2/3))
# img = white_img

params = cv2.SimpleBlobDetector_Params()
params.minThreshold = 30
params.maxThreshold = 60
params.filterByArea = True
# params.minArea = 10
params.maxArea = 50
# params.filterByCircularity = True
# params.minCircularity = 0.2

# # 凸度
# params.filterByConvexity = True
# params.minConvexity = 0.5

# # 慣性比
# params.filterByInertia = True
# params.minInertiaRatio = 0.5
detector = cv2.SimpleBlobDetector_create(params)

keypoints = detector.detect(white_img)
img_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("Keypoints", img_with_keypoints)
cv2.waitKey(0)