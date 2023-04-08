import cv2
import numpy as np
import matplotlib.pyplot as plt


court_img = cv2.imread("badminton_court.jpg")

# Locate points of the documents
# or object which you want to transform
pts1 = np.float32([[0, 0], [971, 0],
                    [0, 1992], [971, 1992]])
pts2 = np.float32([[(0+971)*(1/5), 0], [(0+971)*(4/5), 0],
                    [0, 1992/3], [971, 1992/3]])
    
# Apply Perspective Transform Algorithm
matrix = cv2.getPerspectiveTransform(pts1, pts2)
result = cv2.warpPerspective(court_img, matrix, (971, 664))
    
# # Wrap the transformed image
# cv2.imshow("court_img", court_img) # Initial Capture
# cv2.imshow("result", result) # Transformed Capture
# cv2.waitKey()

alpha = (result[:, :, 1] > 0) * 255
result = np.dstack([result, alpha])
cv2.imwrite("badminton_court_perspective.png", result)