import cv2
import mmcv
import warnings
from tqdm import tqdm

from misc import train_formal_list
from pose_keypoints import pose_keypoints
from data.train_background.classification import img_to_background, background_details
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import inference_top_down_pose_model, init_pose_model, process_mmdet_results, vis_pose_result
from mmpose.datasets import DatasetInfo



TRAIN_DATA_ROOT = "data/train"
OUTPUT_ROOT     = "temp"

MMDETECTION_CONFIG_FILE = "mmdetection/configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person.py"
MMDETECTION_CHECKPOINT  = "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth"
MMPOSE_CONFIG_FILE = "mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_384x288_dark.py"
MMPOSE_CHECKPOINT  = "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_384x288_dark-e881a4b6_20210203.pth"

DEVICE     = "cuda:0"
DET_CAT_ID = 1
BBOX_THR   = 0.3
KPT_THR    = 0.3
RADIUS     = 2
THICKNESS  = 1
RETURN_HEATMAP     = False  # whether to return heatmap, optional
OUTPUT_LAYER_NAMES = None   # return the output of some desired layers, e.g. use ('backbone', ) to return backbone feature

DET_MODEL  = init_detector(MMDETECTION_CONFIG_FILE, MMDETECTION_CHECKPOINT, device=DEVICE)
POSE_MODEL = init_pose_model(MMPOSE_CONFIG_FILE, MMPOSE_CHECKPOINT, device=DEVICE)



def get_player_condition_v1(person_results, court_yt, court_yb):
    player_y = person_results[0]["bbox"][3]
    if abs(player_y-court_yt) < abs(player_y-court_yb):
        return 'A'
    else:
        return 'B'


def get_player_condition_v2(person_results, previous_player_records):
    bbox   = person_results[0]["bbox"]
    A_bbox = previous_player_records['A']["bbox"]
    B_bbox = previous_player_records['B']["bbox"]
    xl,     xr,   yt,   yb =   bbox[0],   bbox[2],   bbox[1],   bbox[3]
    A_xl, A_xr, A_yt, A_yb = A_bbox[0], A_bbox[2], A_bbox[1], A_bbox[3]
    B_xl, B_xr, B_yt, B_yb = B_bbox[0], B_bbox[2], B_bbox[1], B_bbox[3]
    A_diff = (xl-A_xl)**2 + (xr-A_xr)**2 + (yt-A_yt)**2 + (yb-A_yb)**2
    B_diff = (xl-B_xl)**2 + (xr-B_xr)**2 + (yt-B_yt)**2 + (yb-B_yb)**2
    if A_diff < B_diff:
        return 'A'
    else:
        return 'B'


def in_court(person, court_values, condition_values):
    court_xm, court_ym, court_yt, court_yb                 = court_values
    left_ratio, left_constant, right_ratio, right_constant = condition_values
    # Get the inner point
    bbox = person["bbox"]
    bbox_y = bbox[1] if abs(bbox[1]-court_ym) < abs(bbox[3]-court_ym) else bbox[3]
    bbox_x = bbox[0] if abs(bbox[0]-court_xm) < abs(bbox[2]-court_xm) else bbox[2]
    if (court_yt-30 < bbox[1] < court_yb) or \
       (court_yt-30 < bbox[3] < court_yb):                             # Between yt & yb
        if (bbox_x - right_ratio*bbox[1] - right_constant < 0) or \
           (bbox_x - right_ratio*bbox[3] - right_constant < 0):        # On left of right line
            if (bbox_x - left_ratio*bbox[1] - left_constant > 0) or \
               (bbox_x - left_ratio*bbox[3] - left_constant > 0):      # On right of left line
                return True
    return False


def main(video_id):

    dataset = POSE_MODEL.cfg.data["test"]["type"]
    dataset_info = POSE_MODEL.cfg.data["test"].get("dataset_info", None)
    if dataset_info is None:
        warnings.warn("Please set `dataset_info` in the config."
                      "Check https://github.com/open-mmlab/mmpose/pull/663 for details.", DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    video_input_path  = f"data/train/{video_id:05}/{video_id:05}.mp4"
    video_output_path = f"data/train/{video_id:05}/{video_id:05}_pose.mp4"
    csv_output_path = f"data/train/{video_id:05}/{video_id:05}_pose.csv"
    videoReader = mmcv.VideoReader(video_input_path)
    assert videoReader.opened, f"Faild to load video file {video_input_path}"

    fps = videoReader.fps
    size = (videoReader.width, videoReader.height)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    videoWriter = cv2.VideoWriter(video_output_path, fourcc, fps, size)

    # Get 2 players' location
    with open(csv_output_path, mode='w') as csv_file:

        csv_file.write("Frame")
        for player in [ "Player A", "Player B" ]:
            csv_file.write(f",{player} confidence,{player} X left,{player} X right,{player} Y top,{player} Y bottom")
            for kpt in pose_keypoints: csv_file.write(f",{player} {kpt} X,{player} {kpt} Y")
        csv_file.write('\n')

        # Values for checking in-court or not
        bg_id = background_id = img_to_background[video_id]
        bgd   = background_details[bg_id]
        court_yt, court_yb = bgd["y_top"], bgd["y_bottom"]
        court_ym = (court_yt+court_yb) / 2
        court_xlt, court_xlb = bgd["x_left_top"],  bgd["x_left_bottom"]
        court_xrt, court_xrb = bgd["x_right_top"], bgd["x_right_bottom"]
        court_xm = (court_xlt+court_xlb+court_xrt+court_xrb) / 4
        court_lt, court_lb = (court_xlt, court_yt), (court_xlb, court_yb)
        court_rt, court_rb = (court_xrt, court_yt), (court_xrb, court_yb)
        left_ratio    = (court_lt[0]-court_lb[0]) / (court_lt[1]-court_lb[1])
        left_constant = court_lt[0] - left_ratio*court_lt[1]
        right_ratio    = (court_rt[0]-court_rb[0]) / (court_rt[1]-court_rb[1])
        right_constant = court_rt[0] - right_ratio*court_rt[1]
        court_values     = (court_xm, court_ym, court_yt, court_yb)
        condition_values = (left_ratio, left_constant, right_ratio, right_constant)

        previous_player_records = {}
        for frame_id, current_frame in tqdm(enumerate(videoReader), desc=f"[{video_id:05}] Detecting pose"):
            # get the detection results of current frame
            # the resulting box is (x1, y1, x2, y2)
            mmdet_results = inference_detector(DET_MODEL, current_frame)

            # keep the person class bounding boxes.
            person_results = process_mmdet_results(mmdet_results, DET_CAT_ID)

            # width_anchor, height_anchor = videoReader.width/2, videoReader.height*(2/3)
            # person_results = person_results[:5]
            # person_results = sorted(person_results, key=lambda person: (
            #     min(abs(person["bbox"][0]-width_anchor), abs(person["bbox"][2]-width_anchor))**2 *0.5 +
            #     min(abs(person["bbox"][1]-height_anchor), abs(person["bbox"][3]-height_anchor))**2))
            # person_results = sorted(person_results[:2], key=lambda person: person["bbox"][3])

            person_results = list(filter(lambda person: person["bbox"][4] > 0.8, person_results))
            person_results = list(filter(
                lambda person: (person["bbox"][2]-person["bbox"][0])*(person["bbox"][3]-person["bbox"][1]) > 500, person_results))
            person_results = list(filter(
                lambda person: in_court(person, court_values, condition_values), person_results))
            

            # Exceptions
            if (video_id, frame_id) in [ (22, 144), (25, 185), (25, 186), (82, 66) ]:
                person_results = person_results[:2]
            

            if len(person_results) == 2:
                person_results = sorted(person_results, key=lambda person: person["bbox"][3])
            elif len(person_results) == 1:
                # player_condition = get_player_condition_v1(person_results, court_yt, court_yb)
                player_condition = get_player_condition_v2(person_results, previous_player_records)
            elif len(person_results) == 0:
                csv_file.write(f"{frame_id}" + ','*78 + '\n')
                continue
            else:
                print(person_results)
                pose_results, returned_outputs = inference_top_down_pose_model(
                    POSE_MODEL,
                    current_frame,
                    person_results,
                    bbox_thr=BBOX_THR,
                    format="xyxy",
                    dataset=dataset,
                    dataset_info=dataset_info,
                    return_heatmap=RETURN_HEATMAP,
                    outputs=OUTPUT_LAYER_NAMES)
                print(pose_results)
                vis_frame = vis_pose_result(
                    POSE_MODEL,
                    current_frame,
                    pose_results,
                    dataset=dataset,
                    dataset_info=dataset_info,
                    kpt_score_thr=KPT_THR,
                    radius=RADIUS,
                    thickness=THICKNESS,
                    show=False)
                cv2.imshow(f"{video_id:05}.mp4: frame {frame_id}", vis_frame)
                cv2.waitKey(0)
                person_results = person_results[:2]

            pose_results, returned_outputs = inference_top_down_pose_model(
                POSE_MODEL,
                current_frame,
                person_results,
                bbox_thr=BBOX_THR,
                format="xyxy",
                dataset=dataset,
                dataset_info=dataset_info,
                return_heatmap=RETURN_HEATMAP,
                outputs=OUTPUT_LAYER_NAMES)
            
            csv_file.write(f"{frame_id}")
            if len(person_results) == 2:
                previous_player_records['A'] = pose_results[0]
                previous_player_records['B'] = pose_results[1]
                for pose in pose_results:
                    bbox, keypoints = pose["bbox"], pose["keypoints"]
                    csv_file.write(f",{bbox[4]},{bbox[0]},{bbox[2]},{bbox[1]},{bbox[3]}")
                    for kpt in keypoints: csv_file.write(f",{kpt[0]},{kpt[1]}")
            else:
                bbox, keypoints = pose_results[0]["bbox"], pose_results[0]["keypoints"]
                if player_condition == 'A':
                    previous_player_records['A'] = pose_results[0]
                    csv_file.write(f",{bbox[4]},{bbox[0]},{bbox[2]},{bbox[1]},{bbox[3]}")
                    for kpt in keypoints: csv_file.write(f",{kpt[0]},{kpt[1]}")
                    csv_file.write(','*39)
                else:
                    previous_player_records['B'] = pose_results[0]
                    csv_file.write(','*39)
                    csv_file.write(f",{bbox[4]},{bbox[0]},{bbox[2]},{bbox[1]},{bbox[3]}")
                    for kpt in keypoints: csv_file.write(f",{kpt[0]},{kpt[1]}")
            csv_file.write('\n')

            # show the results
            vis_frame = vis_pose_result(
                POSE_MODEL,
                current_frame,
                pose_results,
                dataset=dataset,
                dataset_info=dataset_info,
                kpt_score_thr=KPT_THR,
                radius=RADIUS,
                thickness=THICKNESS,
                show=False)
            
            videoWriter.write(vis_frame)

        videoWriter.release()


if __name__ == '__main__':
    for video_id in train_formal_list:
        if video_id < 82: continue
        main(video_id)