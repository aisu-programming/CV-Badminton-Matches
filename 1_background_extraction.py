import os
import cv2
import mmcv
import winsound
import itertools
import numpy as np
from tqdm import tqdm
from mmdet.apis import inference_detector, init_detector



MODE = "test"

MMDETECTION_CONFIG_FILE = "mmdetection/configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person.py"
MMDETECTION_CHECKPOINT  = "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth"
DET_MODEL = init_detector(MMDETECTION_CONFIG_FILE, MMDETECTION_CHECKPOINT, device="cuda:1")



def average_method(video_id):  # Not good
    os.makedirs(f"outputs/background/avg", exist_ok=True)
    video = np.uint8(mmcv.VideoReader(f"data/train/{video_id:05}/{video_id:05}.mp4"))
    video = video.transpose(1, 2, 0, 3)
    img   = np.uint8(np.average(video, axis=2))
    cv2.imwrite(f"outputs/background/avg/{video_id:05}.png", img)
    return


def mode_method(video_id):  # Good
    os.makedirs(f"outputs/background/mode", exist_ok=True)
    video = np.uint8(mmcv.VideoReader(f"data/train/{video_id:05}/{video_id:05}.mp4"))
    img = video[0]
    width, height = img.shape[:2]
    video = video.transpose(1, 2, 0, 3)
    for w, h in itertools.product(range(width), range(height)):
        uniques, counts = np.unique(video[w, h], axis=0, return_counts=True)
        img[w, h] = uniques[np.argmax(counts)]
    cv2.imwrite(f"outputs/background/mode/{video_id:05}.png", img)
    return


def average_method_with_masked_players(video_id):  # Almost perfect
    os.makedirs(f"outputs/background/avg_without_players/{MODE}", exist_ok=True)
    videoReader = mmcv.VideoReader(f"data/{MODE}/{video_id:05}/{video_id:05}.mp4")
    width, height = videoReader.width, videoReader.height
    video = videoReader[:]

    if MODE=="train":
        if   video_id==678: video = video[36:]
        elif video_id==746: video = video[:-17]

    for frame_id in tqdm(range(len(video)), desc=f"[{MODE}] {video_id:05}.mp4: Extracting background"):
        mmdet_results = inference_detector(DET_MODEL, video[frame_id])[0]
        mmdet_results = sorted(mmdet_results,
                            key=lambda bbox:
                                min(abs(width/2-bbox[0]), abs(width/2-bbox[2]))**2 *0.5 +
                                min(abs(height*(2/3)-bbox[1]), abs(height*(2/3)-bbox[3]))**2 )
        mmdet_results = mmdet_results[:2]
        for bbox in mmdet_results:
            video[frame_id][int(bbox[1])-10:int(bbox[3])+10, int(bbox[0])-10:int(bbox[2])+10] = [ 0, 0, 0 ]
    video = np.uint8(video).transpose(1, 2, 0, 3)
    zero = video==[0,0,0]
    zero = zero[:,:,:,0] * zero[:,:,:,1] * zero[:,:,:,2]
    nonzero = np.logical_not(zero)
    img = np.sum(video, axis=-2)
    img = np.uint8(img/np.expand_dims(np.sum(nonzero, axis=-1)+(1e-9), axis=-1))
    cv2.imwrite(f"outputs/background/avg_without_players/{MODE}/{video_id:05}.png", img)
    return


def mode_method_with_masked_players(video_id):  # ?
    raise NotImplementedError
    os.makedirs(f"outputs/background", exist_ok=True)
    videoReader = mmcv.VideoReader(f"data/train/{video_id:05}/{video_id:05}.mp4")
    width, height = videoReader.width, videoReader.height
    video = videoReader[:]
    for frame_id in tqdm(range(len(video)), desc=f"{video_id:05} - Masking players"):
        mmdet_results = inference_detector(DET_MODEL, video[frame_id])[0]
        mmdet_results = sorted(mmdet_results,
                            key=lambda bbox:
                                    min(abs(width/2-bbox[0]), abs(width/2-bbox[2]))**2 *0.5 +
                                    min(abs(height*(2/3)-bbox[1]), abs(height*(2/3)-bbox[3]))**2 )
        mmdet_results = mmdet_results[:2]
        for bbox in mmdet_results:
            video[frame_id][int(bbox[1])-10:int(bbox[3])+10, int(bbox[0])-10:int(bbox[2])+10] = [ 0, 0, 0 ]
    img = np.ones_like(video[0], dtype=np.uint8)
    video = np.uint8(video).transpose(1, 2, 0, 3)
    print(img.shape, video.shape)
    for w, h in tqdm(itertools.product(range(width), range(height)), total=width*height, desc=f"{video_id:05} - Calculating mode"):
        zero = video[w, h]==[0,0,0]
        zero = zero[:,0] * zero[:,1] * zero[:,2]
        nonzero = np.logical_not(zero)
        if not nonzero.any():
            print(w, h)
            img[w, h] = [0,0,0]
        else:
            uniques, counts = np.unique(video[w, h, nonzero], axis=0, return_counts=True)
            img[w, h] = uniques[ np.argmax(counts) ]
    cv2.imwrite(f"outputs/background/{video_id:05}.png", img)
    return


if __name__ == "__main__":

    assert MODE in ["train", "valid", "test"]
    if   MODE=="train": video_id_list = list(range(1, 800+1))
    elif MODE=="valid": video_id_list = list(range(1, 169+1))
    else              : video_id_list = list(range(170, 399+1))

    for video_id in tqdm(video_id_list):
        # if video_id < 229: continue
        # average_method(video_id)
        # mode_method(video_id)
        average_method_with_masked_players(video_id)
        # mode_method_with_masked_players(video_id)
    
    winsound.Beep(300, 1000)
    pass