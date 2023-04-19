import cv2
import mmcv
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

from misc import train_formal_list
from ball_detection_33 import main as detect_ball


def remove_noise(ball_df:pd.DataFrame):
    for vt_minus in range(10):
        ball_df.loc[ball_df["Smoothed Vis"]<(0.3-vt_minus*0.02), "Visibility"] = 0
        ball_df.loc[ball_df["Smoothed Vis"]<(0.3-vt_minus*0.02), "X"]          = 0
        ball_df.loc[ball_df["Smoothed Vis"]<(0.3-vt_minus*0.02), "Y"]          = 0
        visibility   = ball_df["Visibility"].values
        vis_smoothed = gaussian_filter(np.float32(visibility), sigma=5)
        ball_df = pd.DataFrame({
            "Original Vis": ball_df["Original Vis"],
            "Visibility"  : ball_df["Visibility"],
            "Smoothed Vis": vis_smoothed,
            "X"           : ball_df["X"],
            "Y"           : ball_df["Y"],
            "Time"        : ball_df["Time"],
        })
    return ball_df


def remove_detecting_mistake(ball_df:pd.DataFrame):
    # Find out all detecting mistakes
    vis_ball_df = ball_df[ball_df["Visibility"]==1]
    range_start, range_end = 0, 0
    is_noise_df = pd.DataFrame({"Frame": vis_ball_df.index.values[range_start:range_end]})
    is_noise_df = is_noise_df.set_index("Frame")
    group_id = 0
    while True:
        current_frame = vis_ball_df.index.values[range_end]
        next_frame    = vis_ball_df.index.values[range_end+1]
        range_end += 1
        if next_frame-current_frame>20 or range_end==len(vis_ball_df.values)-1:
            if range_end==len(vis_ball_df.values)-1: range_end += 1
            vbr_x = vis_ball_range = vis_ball_df["X"].values[range_start:range_end]
            vbr_y = vis_ball_range = vis_ball_df["Y"].values[range_start:range_end]
            vbr_x_smoothed = gaussian_filter(np.float32(vbr_x), sigma=3)
            vbr_y_smoothed = gaussian_filter(np.float32(vbr_y), sigma=3)
            #                                          40 / 150
            diff     = np.abs(vbr_x-vbr_x_smoothed)**2*15 + np.abs(vbr_y-vbr_y_smoothed)**2
            is_noise = diff > 25000
            is_noise_df_tmp = pd.DataFrame({
                "Frame"     : vis_ball_df.index.values[range_start:range_end],
                "Smoothed X": vbr_x_smoothed,
                "Smoothed Y": vbr_y_smoothed,
                "Diff"      : diff,
                "is Noise"  : is_noise,
            })
            ball_df.loc[(ball_df.index>=vis_ball_df.index[range_start]) &
                        (ball_df.index<=vis_ball_df.index[range_end-1]), "Group ID"] = group_id
            is_noise_df_tmp = is_noise_df_tmp.set_index("Frame")
            is_noise_df = pd.concat([is_noise_df, is_noise_df_tmp], axis=0)
            if range_end == len(vis_ball_df.values): break
            range_start = range_end
            group_id += 1
    ball_df = pd.concat([ball_df, is_noise_df], axis=1)
    # Remove detecting mistakes
    ball_df.loc[ball_df["is Noise"]==True, "X"] = 0
    ball_df.loc[ball_df["is Noise"]==True, "Y"] = 0
    ball_df = ball_df.rename(columns={"is Noise": "Need Patch"})
    return ball_df


def patch_missing_values(ball_df:pd.DataFrame):
    group_id, group_index = 0, 0
    intepolation_start = None
    group_df = ball_df[ball_df["Group ID"]==group_id]
    while True:
        x_tmp, y_tmp = group_df[["X", "Y"]].values[group_index]
        if intepolation_start is None:
            if x_tmp==0 and y_tmp==0:
                intepolation_start = group_index-1
        else:
            if x_tmp!=0 and y_tmp!=0:
                ivs_x, ivs_y = interpolation_value_start = group_df[["X", "Y"]].values[intepolation_start]
                ive_x, ive_y = interpolation_value_end   = group_df[["X", "Y"]].values[group_index]
                for gi in range(1, group_index-intepolation_start):
                    ball_df.loc[ball_df.index==group_df.index[intepolation_start+gi], "Visibility"] = 1
                    ball_df.loc[ball_df.index==group_df.index[intepolation_start+gi], "X"] = \
                        ivs_x + (ive_x-ivs_x)*(gi/(group_index-intepolation_start))
                    ball_df.loc[ball_df.index==group_df.index[intepolation_start+gi], "Y"] = \
                        ivs_y + (ive_y-ivs_y)*(gi/(group_index-intepolation_start))
                intepolation_start = None
        group_index += 1
        if group_index == len(group_df):
            group_index = 0
            group_id += 1
            group_df = ball_df[ball_df["Group ID"]==group_id]
            if len(group_df)==0: break
    return ball_df


def postprocess(video_id, mode="33"):
    assert mode in ["31", "33"]
    ball_df = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_ball_{mode}.csv", index_col="Frame")
    visibility   = ball_df["Visibility"].values
    vis_smoothed = gaussian_filter(np.float32(visibility), sigma=5)
    ball_df = pd.DataFrame({
        "Original Vis": ball_df["Visibility"],
        "Visibility"  : ball_df["Visibility"],
        "Smoothed Vis": vis_smoothed,
        "X"           : ball_df["X"],
        "Y"           : ball_df["Y"],
        "Time"        : ball_df["Time"],
    })
    ball_df = remove_noise(ball_df)
    ball_df = remove_detecting_mistake(ball_df)
    # ball_df = patch_missing_values(ball_df)
    # ball_df = ball_df.drop(["Smoothed Vis", "Group ID", "Smoothed X", "Smoothed Y", "Diff", "Need Patch"], axis=1)
    ball_df.to_csv(f"data/train/{video_id:05}/{video_id:05}_ball_{mode}_adj.csv")
    return


def output_video(video_id, mode="33"):
    assert mode in ["31", "33"]
    ball_df = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_ball_{mode}_adj.csv")
    videoReader = mmcv.VideoReader(f"data/train/{video_id:05}/{video_id:05}.mp4")
    width, height = videoReader.width, videoReader.height
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = videoReader.fps
    size = (width, height)
    videoWriter = cv2.VideoWriter(f"data/train/{video_id:05}/{video_id:05}_ball_{mode}_adj.mp4", fourcc, fps, size)
    video = np.uint8(videoReader[:])
    for frame, (vxy) in tqdm(zip(video, ball_df[["Visibility", "X", "Y"]].values), desc=f"[{video_id:05}] Saving adjusted video"):
        visibility, ball_x, ball_y = vxy
        if visibility==1: frame = cv2.circle(frame, (round(ball_x), round(ball_y)), 5, (0,0,255), -1)
        videoWriter.write(frame)
    videoWriter.release()
    return


if __name__ == "__main__":
    for video_id in train_formal_list:
        detect_ball(video_id)
        postprocess(video_id, "33")
        # output_video(video_id, "33")
    pass