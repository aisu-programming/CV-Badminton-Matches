import cv2
import mmcv
import numpy as np
# from tqdm import tqdm
# from mmdet.apis import inference_detector, init_detector


videoReader = mmcv.VideoReader("data/train/00001/00001.mp4")
width, height = videoReader.width, videoReader.height
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = videoReader.fps
size = (width, height)
videoWriter = cv2.VideoWriter("test.mp4", fourcc, fps, size)
video = np.uint8(videoReader[:])

# mmdetection_config_file = "mmdetection/configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person.py"
# mmdetection_checkpoint  = "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth"
# det_model = init_detector(mmdetection_config_file, mmdetection_checkpoint, device="cuda:0")
# for frame_id in tqdm(range(len(video)), desc=f"Masking peoples"):
#     mmdet_results = inference_detector(det_model, video[frame_id])[0]
#     mmdet_results = sorted(mmdet_results,
#                         key=lambda bbox:
#                                 min(abs(width/2-bbox[0]), abs(width/2-bbox[2]))**2 *0.5 +
#                                 min(abs(height*(2/3)-bbox[1]), abs(height*(2/3)-bbox[3]))**2 )
#     mmdet_results = mmdet_results[:10]
#     for bbox in mmdet_results:
#         video[frame_id][int(bbox[1])-10:int(bbox[3])+10, int(bbox[0])-10:int(bbox[2])+10] = [ 0, 0, 0 ]

bg = background = cv2.imread("output/background/avg_without_players/type/00000.png")
bg = np.expand_dims(bg, axis=0)
diff = np.sum((video-bg)**2, axis=-1)
video[diff<300] = [0,0,0]

for frame in video: videoWriter.write(frame)
videoWriter.release()