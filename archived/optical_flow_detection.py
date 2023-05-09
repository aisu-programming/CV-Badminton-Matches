# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
from typing import Sequence

import cv2
import numpy as np
from tqdm import tqdm

from mmflow.apis import inference_model, init_model
from mmflow.datasets import visualize_flow


OPTICAL_FLOW_CONFIG_FILE     = "gma_8x2_120k_mixed_368x768.py"
OPTICAL_FLOW_CHECKPOINT_FILE = "gma_8x2_120k_mixed_368x768.pth"
DEVICE                       = "cuda:0"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("video", help="video file")
    parser.add_argument("out",   help="File to save visualized flow map")
    parser.add_argument("--gt",  default=None, help="video file of ground truth for input video")
    args = parser.parse_args()
    return args


def create_video(frames: Sequence[np.ndarray], out: str,
                 fourcc: int, fps: int, size: tuple) -> None:
    """Create a video to save the optical flow.
    Args:
        frames (list, tuple): Image frames.
        out (str): The output file to save visualized flow map.
        fourcc (int): Code of codec used to compress the frames.
        fps (int):      Framerate of the created video stream.
        size (tuple): Size of the video frames.
    """
    # init video writer
    video_writer = cv2.VideoWriter(out, fourcc, fps, size, True)
    for frame in frames:
        video_writer.write(frame)
    video_writer.release()
    return


def main(args) -> None:

    assert args.out[-3:] == "gif" or args.out[-3:] == "mp4", \
        f"Output file must be gif and mp4, but got {args.out[-3:]}."

    # build the model from a config file and a checkpoint file
    model = init_model(OPTICAL_FLOW_CONFIG_FILE, OPTICAL_FLOW_CHECKPOINT_FILE, device=DEVICE)
    # load video
    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened(), f"Failed to load video file {args.video}"
    # get video info
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    imgs = []
    while (cap.isOpened()):
        success, img = cap.read()
        if not success: break
        imgs.append(img)

    frame_list = []
    for i in tqdm(range(len(imgs)-1), desc=f"Calculating optical flow of {args.video}"):
        img1 = imgs[i]
        img2 = imgs[i+1]
        result = inference_model(model, img1, img2)
        # if np.max(result) < 0.2:
        #     if i == 0:
        #         frame = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
        #     else:
        #         frame = frame_list[-1]
        # else:
        #     flow_map = visualize_flow(result, None)
        #     flow_map = cv2.cvtColor(flow_map, cv2.COLOR_RGB2BGR)
        #     frame = flow_map
        img_mask = ((result[:,:,0]**2 + result[:,:,1]**2) **0.5)
        # img_mask = np.minimum(1.0, img_mask*10)
        img_mask = img_mask > 0.1
        img_mask = img_mask[:, :, None]
        frame = np.uint8(img1*img_mask)
        frame_list.append(frame)

    size = (frame_list[0].shape[1], frame_list[0].shape[0])
    cap.release()
    create_video(frame_list, args.out, fourcc, fps, size)
    return


if __name__ == "__main__":
    args = parse_args()
    main(args)