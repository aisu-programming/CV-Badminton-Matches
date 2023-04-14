import os
import cv2
import mmcv
import itertools
import numpy as np
from tqdm import tqdm
from misc import formal_list
from mmdet.apis import inference_detector, init_detector



# # Average method: Not good
# video = np.uint8(mmcv.VideoReader("data/train/00001/00001.mp4"))
# video = video.transpose(1, 2, 0, 3)
# video = np.uint8(np.average(video, axis=2))
# cv2.imshow("", video)
# cv2.waitKey(0)



# # Mode method: Good
# video = np.uint8(mmcv.VideoReader("data/train/00001/00001.mp4"))
# img = video[0]
# width, height = img.shape[:2]
# video = video.transpose(1, 2, 0, 3)
# for w, h in itertools.product(range(width), range(height)):
#     uniques, counts = np.unique(video[w, h], axis=0, return_counts=True)
#     img[w, h] = uniques[np.argmax(counts)]
# cv2.imshow("", img)
# cv2.waitKey(0)



# Average method without player: Almost perfect
os.makedirs(f"output/background/avg_without_players", exist_ok=True)
mmdetection_config_file = "mmdetection/configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person.py"
mmdetection_checkpoint  = "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth"
det_model = init_detector(mmdetection_config_file, mmdetection_checkpoint, device="cuda:0")
for video_id in formal_list:
    videoReader = mmcv.VideoReader(f"data/train/{video_id:05}/{video_id:05}.mp4")
    width, height = videoReader.width, videoReader.height
    video = videoReader[:]
    for frame_id in tqdm(range(len(video)), desc=f"{video_id:05} - Masking players"):
        mmdet_results = inference_detector(det_model, video[frame_id])[0]
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
    cv2.imwrite(f"output/background/avg_without_players/{video_id:05}.png", img)



# # Mode method without player: ?
# os.makedirs(f"output/background", exist_ok=True)
# mmdetection_config_file = "mmdetection/configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person.py"
# mmdetection_checkpoint  = "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth"
# det_model = init_detector(mmdetection_config_file, mmdetection_checkpoint, device="cuda:0")
# for video_id in formal_list:
#     videoReader = mmcv.VideoReader(f"data/train/{video_id:05}/{video_id:05}.mp4")
#     width, height = videoReader.width, videoReader.height
#     video = videoReader[:]
#     for frame_id in tqdm(range(len(video)), desc=f"{video_id:05} - Masking players"):
#         mmdet_results = inference_detector(det_model, video[frame_id])[0]
#         mmdet_results = sorted(mmdet_results,
#                             key=lambda bbox:
#                                     min(abs(width/2-bbox[0]), abs(width/2-bbox[2]))**2 *0.5 +
#                                     min(abs(height*(2/3)-bbox[1]), abs(height*(2/3)-bbox[3]))**2 )
#         mmdet_results = mmdet_results[:2]
#         for bbox in mmdet_results:
#             video[frame_id][int(bbox[1])-10:int(bbox[3])+10, int(bbox[0])-10:int(bbox[2])+10] = [ 0, 0, 0 ]
#     img = np.ones_like(video[0], dtype=np.uint8)
#     video = np.uint8(video).transpose(1, 2, 0, 3)
#     print(img.shape, video.shape)
#     raise NotImplementedError
#     for w, h in tqdm(itertools.product(range(width), range(height)), total=width*height, desc=f"{video_id:05} - Calculating mode"):
#         zero = video[w, h]==[0,0,0]
#         zero = zero[:,0] * zero[:,1] * zero[:,2]
#         nonzero = np.logical_not(zero)
#         if not nonzero.any():
#             print(w, h)
#             img[w, h] = [0,0,0]
#         else:
#             uniques, counts = np.unique(video[w, h, nonzero], axis=0, return_counts=True)
#             img[w, h] = uniques[ np.argmax(counts) ]
#     cv2.imwrite(f"output/background/{video_id:05}.png", img)