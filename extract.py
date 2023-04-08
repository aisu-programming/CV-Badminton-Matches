import cv2
import os
  
# Read the video from specified path
video = cv2.VideoCapture("data/train/00001/00001.mp4")
  
# frame
currentframe = 0

os.makedirs("data/train/00001/images", exist_ok=True)

while(True):
      
    # reading from frame
    ret, frame = video.read()
  
    if ret:
        # if video is still left continue creating images
        name = f"data/train/00001/images/{currentframe:05d}.png"
        # print ("Creating..." + name)
  
        # writing the extracted images
        cv2.imwrite(name, frame)
  
        # increasing counter so that it will
        # show how many frames are created
        currentframe += 1
    else:
        break
  
# Release all space and windows once done
video.release()
cv2.destroyAllWindows()