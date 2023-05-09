import os
# import sys
import cv2
import time
# import getopt
# import piexif
import numpy as np
# import pandas as pd
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
# import tensorflow as tf
import keras.backend as K
# from PIL import Image
# from glob import glob
from tqdm import tqdm
# from os.path import isfile, join
# from keras import optimizers
from keras.models import *
from keras.layers import *
# from sklearn.model_selection import train_test_split

# from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.utils import array_to_img, img_to_array  # , load_img
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# from TrackNetv2 import TrackNet3
from misc import train_formal_list, valid_formal_list


BATCH_SIZE=1
HEIGHT=288
WIDTH=512
#HEIGHT=360
#WIDTH=640
sigma=2.5
mag=1


def genHeatMap(w, h, cx, cy, r, mag):
	if cx < 0 or cy < 0:
		return np.zeros((h, w))
	x, y = np.meshgrid(np.linspace(1, w, w), np.linspace(1, h, h))
	heatmap = ((y - (cy + 1))**2) + ((x - (cx + 1))**2)
	heatmap[heatmap <= r**2] = 1
	heatmap[heatmap > r**2] = 0
	return heatmap*mag

#time: in milliseconds
def custom_time(time):
	remain = int(time / 1000)
	ms = (time / 1000) - remain
	s = remain % 60
	s += ms
	remain = int(remain / 60)
	m = remain % 60
	remain = int(remain / 60)
	h = remain
	#Generate custom time string
	cts = ''
	if len(str(h)) >= 2:
		cts += str(h)
	else:
		for i in range(2 - len(str(h))):
			cts += '0'
		cts += str(h)
	
	cts += ':'

	if len(str(m)) >= 2:
		cts += str(m)
	else:
		for i in range(2 - len(str(m))):
			cts += '0'
		cts += str(m)

	cts += ':'

	if len(str(int(s))) == 1:
		cts += '0'
	cts += str(s)

	return cts


def main(video_id, mode):

	video_name = f"data/{mode}/{video_id:05}/{video_id:05}.mp4"
	load_weights = "TrackNetv2/mimo/model906_30"

	# Loss function
	def custom_loss(y_true, y_pred):
		loss = (-1)*(K.square(1 - y_pred) * y_true * K.log(K.clip(y_pred, K.epsilon(), 1)) + K.square(y_pred) * (1 - y_true) * K.log(K.clip(1 - y_pred, K.epsilon(), 1)))
		return K.mean(loss)

	model = load_model(load_weights, custom_objects={'custom_loss':custom_loss})
	# model.summary()
	# start = time.time()

	f = open(video_name[:-4]+'_ball_33.csv', 'w')
	f.write('Frame,Visibility,X,Y,Time\n')

	cap = cv2.VideoCapture(video_name)

	success, image1 = cap.read()
	frame_time1 = custom_time(cap.get(cv2.CAP_PROP_POS_MSEC))
	success, image2 = cap.read()
	frame_time2 = custom_time(cap.get(cv2.CAP_PROP_POS_MSEC))
	success, image3 = cap.read()
	frame_time3 = custom_time(cap.get(cv2.CAP_PROP_POS_MSEC))

	ratio = image1.shape[0] / HEIGHT

	size = (int(WIDTH*ratio), int(HEIGHT*ratio))
	fps = 30

	if video_name[-3:] == 'avi':
		fourcc = cv2.VideoWriter_fourcc(*'DIVX')
	elif video_name[-3:] == 'mp4':
		fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	else:
		print('usage: video type can only be .avi or .mp4')
		exit(1)

	video_writer = cv2.VideoWriter(video_name[:-4]+'_ball_33'+video_name[-4:], fourcc, fps, size)

	count = 0
	
	pbar = tqdm(desc=f"[{video_id:05}] Predicting ball")
	while success:
		unit = []
		#Adjust BGR format (cv2) to RGB format (PIL)
		x1 = image1[...,::-1]
		x2 = image2[...,::-1]
		x3 = image3[...,::-1]
		#Convert np arrays to PIL images
		x1 = array_to_img(x1)
		x2 = array_to_img(x2)
		x3 = array_to_img(x3)
		#Resize the images
		x1 = x1.resize(size = (WIDTH, HEIGHT))
		x2 = x2.resize(size = (WIDTH, HEIGHT))
		x3 = x3.resize(size = (WIDTH, HEIGHT))
		#Convert images to np arrays and adjust to channels first
		x1 = np.moveaxis(img_to_array(x1), -1, 0)
		x2 = np.moveaxis(img_to_array(x2), -1, 0)
		x3 = np.moveaxis(img_to_array(x3), -1, 0)
		#Create data
		unit.append(x1[0])
		unit.append(x1[1])
		unit.append(x1[2])
		unit.append(x2[0])
		unit.append(x2[1])
		unit.append(x2[2])
		unit.append(x3[0])
		unit.append(x3[1])
		unit.append(x3[2])
		unit=np.asarray(unit)	
		unit = unit.reshape((1, 9, HEIGHT, WIDTH))
		unit = unit.astype('float32')
		unit /= 255
		y_pred = model.predict(unit, batch_size=BATCH_SIZE, verbose=0)
		y_pred = y_pred > 0.5
		y_pred = y_pred.astype('float32')
		h_pred = y_pred[0]*255
		h_pred = h_pred.astype('uint8')
		for i in range(3):
			if i == 0:
				frame_time = frame_time1
				image = image1
			elif i == 1:
				frame_time = frame_time2
				image = image2
			elif i == 2:
				frame_time = frame_time3
				image = image3

			if np.amax(h_pred[i]) <= 0:
				f.write(str(count)+',0,0,0,'+frame_time+'\n')
				# video_writer.write(image)
			else:	
				#h_pred
				(cnts, _) = cv2.findContours(h_pred[i].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
				rects = [cv2.boundingRect(ctr) for ctr in cnts]
				max_area_idx = 0
				max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
				for i in range(len(rects)):
					area = rects[i][2] * rects[i][3]
					if area > max_area:
						max_area_idx = i
						max_area = area
				target = rects[max_area_idx]
				(cx_pred, cy_pred) = (int(ratio*(target[0] + target[2] / 2)), int(ratio*(target[1] + target[3] / 2)))

				f.write(str(count)+',1,'+str(cx_pred)+','+str(cy_pred)+','+frame_time+'\n')
				image_cp = np.copy(image)
				cv2.circle(image_cp, (cx_pred, cy_pred), 5, (0,0,255), -1)
				# video_writer.write(image_cp)
			count += 1
		success, image1 = cap.read()
		frame_time1 = custom_time(cap.get(cv2.CAP_PROP_POS_MSEC))
		success, image2 = cap.read()
		frame_time2 = custom_time(cap.get(cv2.CAP_PROP_POS_MSEC))
		success, image3 = cap.read()
		frame_time3 = custom_time(cap.get(cv2.CAP_PROP_POS_MSEC))
		pbar.update(3)

	f.close()
	# video_writer.release()
	# end = time.time()
	# print('Prediction time:', end-start, 'secs')
	return


if __name__ == "__main__":

    MODE = "valid"
    assert MODE in [ "train", "valid" ]

    if MODE=="train": video_id_list = train_formal_list
    else            : video_id_list = valid_formal_list

    for video_id in video_id_list: main(video_id, MODE)