import os
import warnings
from argparse import ArgumentParser

import cv2
import mmcv

from mmpose.apis import (collect_multi_frames, inference_top_down_pose_model,
                         init_pose_model, process_mmdet_results,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


TRAIN_DATA_ROOT = "data/train"
OUTPUT_ROOT     = "temp"


def main():
    """Visualize the demo video (support both single-frame and multi-frame).

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument("--train-data-root", type=str, default=TRAIN_DATA_ROOT, help="")
    parser.add_argument("--output-root", type=str, default=OUTPUT_ROOT, help="")
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument("--det-cat-id",
        type=int, default=1,  # person
        help="Category id for bounding box detection model")
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--radius',
        type=int,
        default=2,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')
    # parser.add_argument("--use-multi-frames",
    #     action="store_true", default=False,
    #     help="whether to use multi frames for inference in the pose"
    #     "estimation stage. Default: False.")

    assert has_mmdet, "Please install mmdet to run the demo."

    args = parser.parse_args()

    mmdetection_config_file = "mmdetection/configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person.py"
    mmdetection_checkpoint  = "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth"
    mmpose_config_file = "mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_384x288_dark.py"
    mmpose_checkpoint  = "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_384x288_dark-e881a4b6_20210203.pth"

    det_model  = init_detector(mmdetection_config_file, mmdetection_checkpoint, device=args.device.lower())
    pose_model = init_pose_model(mmpose_config_file, mmpose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data["test"]["type"]
    dataset_info = pose_model.cfg.data["test"].get("dataset_info", None)
    if dataset_info is None:
        warnings.warn(
            "Please set `dataset_info` in the config."
            "Check https://github.com/open-mmlab/mmpose/pull/663 for details.",
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    for video_id in os.listdir(args.train_data_root):
        video_path = f"{args.train_data_root}/{video_id}/{video_id}.mp4"
        video = mmcv.VideoReader(video_path)
        assert video.opened, f"Faild to load video file {video_path}"

        if args.output_root == '':
            save_out_video = False
        else:
            os.makedirs(args.output_root, exist_ok=True)
            save_out_video = True

        if save_out_video:
            fps = video.fps
            size = (video.width, video.height)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            videoWriter = cv2.VideoWriter(
                os.path.join(
                    args.output_root, f"{os.path.basename(video_path)}"
                ), fourcc, fps, size
            )

        # # frame index offsets for inference, used in multi-frame inference setting
        # if args.use_multi_frames:
        #     assert "frame_indices_test" in pose_model.cfg.data.test.data_cfg
        #     indices = pose_model.cfg.data.test.data_cfg["frame_indices_test"]

        # whether to return heatmap, optional
        return_heatmap = False

        # return the output of some desired layers,
        # e.g. use ('backbone', ) to return backbone feature
        output_layer_names = None

        # Get 2 players' location
        for frame_id, current_frame in enumerate(mmcv.track_iter_progress(video)):
            # get the detection results of current frame
            # the resulting box is (x1, y1, x2, y2)
            mmdet_results = inference_detector(det_model, current_frame)

            # keep the person class bounding boxes.
            person_results = process_mmdet_results(mmdet_results, args.det_cat_id)

            # if args.use_multi_frames:
            #     frames = collect_multi_frames(video, frame_id, indices, args.online)

            width_anchor, height_anchor = video.width/2, video.height*2/3
            person_results = person_results[:5]
            person_results = sorted(person_results, key=lambda person: (
                min(abs(person['bbox'][0]-width_anchor), abs(person['bbox'][2]-width_anchor))**2 +
                min(abs(person['bbox'][1]-height_anchor), abs(person['bbox'][3]-height_anchor))**2*0.8))
            person_results = person_results[:2]
            if frame_id == 0:
                person_results_last_frame = person_results
            else:
                person_results_verified = []
                for person_lf in person_results_last_frame:
                    match_flag = False
                    person_lf_width_center = (person_lf['bbox'][0] + person_lf['bbox'][2]) / 2
                    for person in person_results:
                        person_width_center = (person['bbox'][0] + person['bbox'][2]) / 2
                        if abs(person_lf_width_center-person_width_center) < 50 and abs(person_lf['bbox'][3]-person['bbox'][3]) < 50:
                            person_results_verified.append(person)
                            match_flag = True
                            break
                    if not match_flag:
                        person_results_verified.append(person_lf)
                person_results = person_results_verified
                person_results_last_frame = person_results

        # Get 2 players' location
        for frame_id, current_frame in enumerate(mmcv.track_iter_progress(video)):
            # get the detection results of current frame
            # the resulting box is (x1, y1, x2, y2)
            mmdet_results = inference_detector(det_model, current_frame)

            # keep the person class bounding boxes.
            person_results = process_mmdet_results(mmdet_results, args.det_cat_id)

            width_anchor, height_anchor = video.width/2, video.height*2/3
            person_results = person_results[:5]
            person_results = sorted(person_results, key=lambda person: (
                min(abs(person['bbox'][0]-width_anchor), abs(person['bbox'][2]-width_anchor))**2 +
                min(abs(person['bbox'][1]-height_anchor), abs(person['bbox'][3]-height_anchor))**2*0.5))
            person_results = sorted(person_results[:2], key=lambda person: person['bbox'][3])
            player_A, player_B = person_results[0], person_results[1]

        for frame_id, current_frame in enumerate(zip(mmcv.track_iter_progress(video), person_results)):
            pose_results, returned_outputs = inference_top_down_pose_model(
                pose_model,
                current_frame,  # frames if args.use_multi_frames else current_frame,
                person_results,
                bbox_thr=args.bbox_thr,
                format="xyxy",
                dataset=dataset,
                dataset_info=dataset_info,
                return_heatmap=return_heatmap,
                outputs=output_layer_names)

        # show the results
        for frame_id, current_frame in enumerate(mmcv.track_iter_progress(video)):
            vis_frame = vis_pose_result(
                pose_model,
                current_frame,
                pose_results,
                dataset=dataset,
                dataset_info=dataset_info,
                kpt_score_thr=args.kpt_thr,
                radius=args.radius,
                thickness=args.thickness,
                show=False)
            
            if save_out_video:
                videoWriter.write(vis_frame)

        if save_out_video:
            videoWriter.release()


if __name__ == '__main__':
    main()
