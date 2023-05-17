""" Libraries """
# import os
import torch
import argparse
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from tqdm import tqdm
# from scipy.signal import find_peaks
# from scipy.ndimage import gaussian_filter1d

from libs.dataloader import draw_limbs  # , draw_kpts
from data.background.classification import train_img_to_background, valid_img_to_background, test_img_to_background



""" Functions """
def main(args):

    # os.makedirs(args.save_dir, exist_ok=True)

    model = torch.load(args.model_path).to(args.device)
    model = model.eval()

    if   args.mode=="train": video_id_list = list(range(1, 800+1))
    elif args.mode=="valid": video_id_list = list(range(1, 169+1))
    else                   : video_id_list = list(range(170, 399+1))

    pbar = tqdm(video_id_list)
    for video_id in pbar:
        pbar.set_description(f"[{args.mode}] {video_id:05}.mp4 - Predicting Winner")

        # if video_id > 50: continue

        answer_df_values  = pd.read_csv(f"data/{args.mode}/{video_id:05}/{video_id:05}_prediction_1_hitter.csv")[["HitFrame", "Hitter"]].values
        ball_df_values    = pd.read_csv(f"data/{args.mode}/{video_id:05}/{video_id:05}_ball_33_adj.csv")[["Adjusted X", "Adjusted Y"]].values
        pose_df_values    = pd.read_csv(f"data/{args.mode}/{video_id:05}/{video_id:05}_pose_wholebody.csv").values
        video_frame_count = len(pose_df_values)

        length = args.length
        ll = 15             # left_length
        rl = (length-1)-ll  # right_length

        input_imgs, input_kpts, input_balls, input_bg_ids, input_hitters, input_times = [], [], [], [], [], []
        hit_frame, hitter = answer_df_values[-1]
        hf_start, hf_end = max(hit_frame-ll, 0), min(hit_frame+rl, video_frame_count-1)

        A_kpts_ori_xs = pose_df_values[hf_start:hf_end+1, (  6  ):(  6+266):2] / 640 - 1.0
        A_kpts_ori_ys = pose_df_values[hf_start:hf_end+1, (  6+1):(  6+266):2] / 360 - 1.0
        B_kpts_ori_xs = pose_df_values[hf_start:hf_end+1, (277  ):(277+266):2] / 640 - 1.0
        B_kpts_ori_ys = pose_df_values[hf_start:hf_end+1, (277+1):(277+266):2] / 360 - 1.0
        
        A_kpts_scl_xs = pose_df_values[hf_start:hf_end+1, (  6  ):(  6+266):2]
        A_kpts_scl_ys = pose_df_values[hf_start:hf_end+1, (  6+1):(  6+266):2]
        B_kpts_scl_xs = pose_df_values[hf_start:hf_end+1, (277  ):(277+266):2]
        B_kpts_scl_ys = pose_df_values[hf_start:hf_end+1, (277+1):(277+266):2]

        if not np.isnan(A_kpts_scl_xs).all():
            A_kpts_scl_xs = A_kpts_scl_xs - (np.nanmax(A_kpts_scl_xs) + np.nanmin(A_kpts_scl_xs)) / 2
        if not np.isnan(A_kpts_scl_ys).all():
            A_kpts_scl_ys = A_kpts_scl_ys - (np.nanmax(A_kpts_scl_ys) + np.nanmin(A_kpts_scl_ys)) / 2
        if not np.isnan(B_kpts_scl_xs).all():
            B_kpts_scl_xs = B_kpts_scl_xs - (np.nanmax(B_kpts_scl_xs) + np.nanmin(B_kpts_scl_xs)) / 2
        if not np.isnan(B_kpts_scl_ys).all():
            B_kpts_scl_ys = B_kpts_scl_ys - (np.nanmax(B_kpts_scl_ys) + np.nanmin(B_kpts_scl_ys)) / 2

        if not np.isnan(A_kpts_scl_xs).all() or not np.isnan(A_kpts_scl_ys).all():
            A_kpts_scl_xs /= np.nanmax([np.nanmax(np.abs(A_kpts_scl_xs)), np.nanmax(np.abs(A_kpts_scl_ys))])
            A_kpts_scl_ys /= np.nanmax([np.nanmax(np.abs(A_kpts_scl_xs)), np.nanmax(np.abs(A_kpts_scl_ys))])
        if not np.isnan(B_kpts_scl_xs).all() or not np.isnan(B_kpts_scl_ys).all():
            B_kpts_scl_xs /= np.nanmax([np.nanmax(np.abs(B_kpts_scl_xs)), np.nanmax(np.abs(B_kpts_scl_ys))])
            B_kpts_scl_ys /= np.nanmax([np.nanmax(np.abs(B_kpts_scl_xs)), np.nanmax(np.abs(B_kpts_scl_ys))])

        ball_datas = ball_df_values[hf_start:hf_end+1]
        ball_datas[:, 0] = ball_datas[:, 0] / 640 - 1.0
        ball_datas[:, 1] = ball_datas[:, 1] / 360 - 1.0
        ball_datas = np.nan_to_num(ball_datas, nan=0.0)

        if hit_frame-ll < 0:
            A_kpts_ori_xs = np.concatenate([ np.zeros((abs(hit_frame-ll), 133)), A_kpts_ori_xs ], axis=0)
            A_kpts_ori_ys = np.concatenate([ np.zeros((abs(hit_frame-ll), 133)), A_kpts_ori_ys ], axis=0)
            B_kpts_ori_xs = np.concatenate([ np.zeros((abs(hit_frame-ll), 133)), B_kpts_ori_xs ], axis=0)
            B_kpts_ori_ys = np.concatenate([ np.zeros((abs(hit_frame-ll), 133)), B_kpts_ori_ys ], axis=0)
            A_kpts_scl_xs = np.concatenate([ np.zeros((abs(hit_frame-ll), 133)), A_kpts_scl_xs ], axis=0)
            A_kpts_scl_ys = np.concatenate([ np.zeros((abs(hit_frame-ll), 133)), A_kpts_scl_ys ], axis=0)
            B_kpts_scl_xs = np.concatenate([ np.zeros((abs(hit_frame-ll), 133)), B_kpts_scl_xs ], axis=0)
            B_kpts_scl_ys = np.concatenate([ np.zeros((abs(hit_frame-ll), 133)), B_kpts_scl_ys ], axis=0)
            ball_datas    = np.concatenate([ np.zeros((abs(hit_frame-ll),   2)), ball_datas    ], axis=0)
        if hit_frame+rl > video_frame_count-1:
            A_kpts_ori_xs = np.concatenate([ A_kpts_ori_xs, np.zeros((hit_frame+rl-(video_frame_count-1), 133)) ], axis=0)
            A_kpts_ori_ys = np.concatenate([ A_kpts_ori_ys, np.zeros((hit_frame+rl-(video_frame_count-1), 133)) ], axis=0)
            B_kpts_ori_xs = np.concatenate([ B_kpts_ori_xs, np.zeros((hit_frame+rl-(video_frame_count-1), 133)) ], axis=0)
            B_kpts_ori_ys = np.concatenate([ B_kpts_ori_ys, np.zeros((hit_frame+rl-(video_frame_count-1), 133)) ], axis=0)
            A_kpts_scl_xs = np.concatenate([ A_kpts_scl_xs, np.zeros((hit_frame+rl-(video_frame_count-1), 133)) ], axis=0)
            A_kpts_scl_ys = np.concatenate([ A_kpts_scl_ys, np.zeros((hit_frame+rl-(video_frame_count-1), 133)) ], axis=0)
            B_kpts_scl_xs = np.concatenate([ B_kpts_scl_xs, np.zeros((hit_frame+rl-(video_frame_count-1), 133)) ], axis=0)
            B_kpts_scl_ys = np.concatenate([ B_kpts_scl_ys, np.zeros((hit_frame+rl-(video_frame_count-1), 133)) ], axis=0)
            ball_datas    = np.concatenate([ ball_datas,    np.zeros((hit_frame+rl-(video_frame_count-1),   2)) ], axis=0)
        
        assert ball_datas.shape == (length, 2)

        kpt_imgs = np.zeros((2, length, 64, 64))
        for fid, hf in enumerate(range(hit_frame-ll, hit_frame+rl+1)):
            if hf <                  0: continue
            if hf >= video_frame_count: continue
            if pose_df_values[hf, 1] > 0.5:
                kpts_x = A_kpts_scl_xs[fid, :17]
                kpts_y = A_kpts_scl_ys[fid, :17]
                kpts_x = np.array(((kpts_x/2)+0.5) *60 +2, dtype=np.uint8).tolist()
                kpts_y = np.array(((kpts_y/2)+0.5) *60 +2, dtype=np.uint8).tolist()
                kpt_imgs[0, fid] = draw_limbs(kpts_x, kpts_y)
            if pose_df_values[hf, 272] > 0.5:
                kpts_x = B_kpts_scl_xs[fid, :17]
                kpts_y = B_kpts_scl_ys[fid, :17]
                kpts_x = np.array(((kpts_x/2)+0.5) *60 +2, dtype=np.uint8).tolist()
                kpts_y = np.array(((kpts_y/2)+0.5) *60 +2, dtype=np.uint8).tolist()
                kpt_imgs[1, fid] = draw_limbs(kpts_x, kpts_y)

        kpt_datas = np.concatenate([
            np.expand_dims(np.nan_to_num(A_kpts_ori_xs, nan=0.0), axis=1),
            np.expand_dims(np.nan_to_num(A_kpts_ori_ys, nan=0.0), axis=1),
            np.expand_dims(np.nan_to_num(B_kpts_ori_xs, nan=0.0), axis=1),
            np.expand_dims(np.nan_to_num(B_kpts_ori_ys, nan=0.0), axis=1),
        ], axis=1)
        assert kpt_datas.shape == (length, 4, 133)

        time_datas = np.arange(hit_frame-ll, hit_frame+rl+1) / video_frame_count
        assert time_datas.shape == (length, )

        bg_id = np.zeros(13)
        if   args.mode=="train": bg_id[train_img_to_background[video_id]] = 1
        elif args.mode=="valid": bg_id[valid_img_to_background[video_id]] = 1
        else:
            if test_img_to_background[video_id] == 13:
                bg_id[11] = 1
            else:
                bg_id[test_img_to_background[video_id]] = 1

        hitter = [ 1, 0 ] if hitter=='A' else [ 0, 1 ]
        
        input_imgs.append(kpt_imgs)
        input_kpts.append(kpt_datas)
        input_balls.append(ball_datas)
        input_times.append(time_datas)
        input_bg_ids.append(bg_id)
        input_hitters.append(hitter)

        input_imgs    = torch.from_numpy(np.array(input_imgs,    dtype=np.float32)).to(args.device)
        input_kpts    = torch.from_numpy(np.array(input_kpts,    dtype=np.float32)).to(args.device)
        input_balls   = torch.from_numpy(np.array(input_balls,   dtype=np.float32)).to(args.device)
        input_times   = torch.from_numpy(np.array(input_times,   dtype=np.float32)).to(args.device)
        input_bg_ids  = torch.from_numpy(np.array(input_bg_ids,  dtype=np.float32)).to(args.device)
        input_hitters = torch.from_numpy(np.array(input_hitters, dtype=np.float32)).to(args.device)
        prediction = model(input_imgs, input_kpts, input_balls, input_times, input_bg_ids, input_hitters)
        
        prediction = np.array(prediction.cpu().detach().numpy()[0])
        assert prediction.shape == (2,)
        prediction = np.argmax(prediction, axis=-1)

        winner = ['X'] *(len(answer_df_values)-1)
        winner = winner + ['A'] if prediction==0 else winner + ['B']

        output_df = pd.read_csv(f"data/{args.mode}/{video_id:05}/{video_id:05}_prediction_8_ball_type.csv")
        output_df["Winner"] = winner
        output_df = output_df.set_index("ShotSeq")
        output_df.to_csv(f"data/{args.mode}/{video_id:05}/{video_id:05}_prediction_9_winner.csv")

    return



""" Execution """
if __name__ == "__main__":

    LOAD_DIR           = "2023.05.14-20.38.24"
    DEFAULT_MODE       = "valid"
    DEFAULT_BATCH_SIZE = 200
    DEFAULT_LENGTH     = 121
    DEFAULT_DEVICE     = "cuda:1"
    DEFAULT_MODEL_PATH = f"logs/all/9_winner/{LOAD_DIR}/best_valid_loss.pt"

    parser = argparse.ArgumentParser()
    parser.add_argument("-m",  "--mode",       type=str, default=DEFAULT_MODE)
    parser.add_argument("-bs", "--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("-l",  "--length",     type=int, default=DEFAULT_LENGTH)
    parser.add_argument("-d",  "--device",     type=str, default=DEFAULT_DEVICE)
    parser.add_argument("-mp", "--model-path", type=str, default=DEFAULT_MODEL_PATH)

    args = parser.parse_args()
    assert args.mode in [ "train", "valid", "test" ]
    args.save_dir = f"predictions/9_winner/{LOAD_DIR}_{args.mode}"
    main(args)