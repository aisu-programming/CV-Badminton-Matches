# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import numpy as np
import tempfile
from argparse import ArgumentParser

import mmcv

from mmtrack.apis import inference_mot, init_model


CONFIG     = "mmtracking/configs/mot/bytetrack/bytetrack_yolox_x_crowdhuman_mot17-private-half.py"
CHECKPOINT = "https://download.openmmlab.com/mmtracking/mot/bytetrack/bytetrack_yolox_x/bytetrack_yolox_x_crowdhuman_mot17-private-half_20211218_205500-1985c9f0.pth"


def main():
    parser = ArgumentParser()
    parser.add_argument("--score-thr", type=float, default=0.0, help="The threshold of score to filter bboxes.")
    parser.add_argument("--device", default="cuda:0", help="device used for inference")
    parser.add_argument("--backend", choices=["cv2", "plt"], default="cv2", help="the backend to visualize the results")
    parser.add_argument("--fps", help="FPS of the output video")
    args = parser.parse_args()

    for video_id in os.listdir("data/train/"):

        args.input  = f"data/train/{video_id}/{video_id}.mp4"
        args.output = f"mot_ouput/{video_id}.mp4"

        video = mmcv.VideoReader(args.input)
        IN_VIDEO = True
        # define output
        if args.output is not None:
            if args.output.endswith('.mp4'):
                OUT_VIDEO = True
                out_dir = tempfile.TemporaryDirectory()
                out_path = out_dir.name
                _out = args.output.rsplit(os.sep, 1)
                if len(_out) > 1:
                    os.makedirs(_out[0], exist_ok=True)
            else:
                OUT_VIDEO = False
                out_path = args.output
                os.makedirs(out_path, exist_ok=True)

        fps = args.fps
        if OUT_VIDEO:
            if fps is None and IN_VIDEO:
                fps = video.fps
            if not fps:
                raise ValueError('Please set the FPS for the output video.')
            fps = int(fps)

        # build the model from a config file and a checkpoint file
        model = init_model(CONFIG, CHECKPOINT, device=args.device)

        prog_bar = mmcv.ProgressBar(len(video))
        # test and show/save the images
        
        for i, img in enumerate(video):
            if i >= 30: break
            if isinstance(img, str):
                img = osp.join(args.input, img)
            result = inference_mot(model, img, frame_id=i)

            width_anchor, height_anchor = video.width/2, video.height*2/3
            result["track_bboxes"] = [np.array(list(sorted(result["track_bboxes"][0], key=lambda bbox: (
                min(abs(bbox[1]-width_anchor), abs(bbox[3]-width_anchor))**2 +
                min(abs(bbox[2]-height_anchor), abs(bbox[4]-height_anchor))**2*0.5))[:2]))]

            if args.output is not None:
                if IN_VIDEO or OUT_VIDEO:
                    out_file = osp.join(out_path, f'{i:06d}.jpg')
                else:
                    out_file = osp.join(out_path, img.rsplit(os.sep, 1)[-1])
            else:
                out_file = None
            model.show_result(
                img,
                result,
                score_thr=args.score_thr,
                show=False,
                wait_time=int(1000. / fps) if fps else 0,
                out_file=out_file,
                backend=args.backend)
            prog_bar.update()

        if args.output and OUT_VIDEO:
            print(f'making the output video at {args.output} with a FPS of {fps}')
            mmcv.frames2video(out_path, args.output, fps=fps, fourcc='mp4v')
            out_dir.cleanup()


if __name__ == '__main__':
    main()
