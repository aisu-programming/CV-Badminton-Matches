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
from libs.misc import train_formal_list, valid_formal_list
from data.train_background.classification import img_to_background as train_img_to_background
from data.valid_background.classification import img_to_background as valid_img_to_background


def main(args):

    # os.makedirs(args.save_dir, exist_ok=True)

    model = torch.load(args.model_path).to(args.device)
    model = model.eval()

    if args.mode=="train": video_id_list = train_formal_list
    else                 : video_id_list = valid_formal_list

    for video_id in video_id_list:

        # if video_id > 50: continue

        answer_df_values  = pd.read_csv(f"data/{args.mode}/{video_id:05}/{video_id:05}_prediction_2_round_head.csv")[["HitFrame", "Hitter", "RoundHead"]].values
        ball_df_values    = pd.read_csv(f"data/{args.mode}/{video_id:05}/{video_id:05}_ball_33_adj.csv")[["Adjusted X", "Adjusted Y"]].values
        pose_df_values    = pd.read_csv(f"data/{args.mode}/{video_id:05}/{video_id:05}_pose.csv").values
        video_frame_count = len(pose_df_values)
        hl = (args.length-1) // 2

        predictions = []
        counter, input_imgs, input_kpts, input_balls, input_bg_ids, input_hitters, input_times = 0, [], [], [], [], [], []
        for hid, (hit_frame, hitter, _) in tqdm(enumerate(answer_df_values), desc=f"{video_id:05}.mp4", total=len(answer_df_values)):

            hf_start, hf_end = max(hit_frame-hl, 0), min(hit_frame+hl, video_frame_count-1)
            A_kpts_ori_xs = pose_df_values[hf_start:hf_end+1, ( 6  ):( 6+34):2] / 640 - 1.0
            A_kpts_ori_ys = pose_df_values[hf_start:hf_end+1, ( 6+1):( 6+34):2] / 360 - 1.0
            B_kpts_ori_xs = pose_df_values[hf_start:hf_end+1, (45  ):(45+34):2] / 640 - 1.0
            B_kpts_ori_ys = pose_df_values[hf_start:hf_end+1, (45+1):(45+34):2] / 360 - 1.0
            
            A_kpts_scl_xs = pose_df_values[hf_start:hf_end+1, ( 6  ):( 6+34):2]
            A_kpts_scl_ys = pose_df_values[hf_start:hf_end+1, ( 6+1):( 6+34):2]
            B_kpts_scl_xs = pose_df_values[hf_start:hf_end+1, (45  ):(45+34):2]
            B_kpts_scl_ys = pose_df_values[hf_start:hf_end+1, (45+1):(45+34):2]
            if np.isnan(A_kpts_scl_xs).all() or np.isnan(A_kpts_scl_ys).all() or \
                np.isnan(B_kpts_scl_xs).all() or np.isnan(B_kpts_scl_ys).all(): continue
            A_kpts_scl_xs = A_kpts_scl_xs - (np.nanmax(A_kpts_scl_xs) + np.nanmin(A_kpts_scl_xs)) / 2
            A_kpts_scl_ys = A_kpts_scl_ys - (np.nanmax(A_kpts_scl_ys) + np.nanmin(A_kpts_scl_ys)) / 2
            B_kpts_scl_xs = B_kpts_scl_xs - (np.nanmax(B_kpts_scl_xs) + np.nanmin(B_kpts_scl_xs)) / 2
            B_kpts_scl_ys = B_kpts_scl_ys - (np.nanmax(B_kpts_scl_ys) + np.nanmin(B_kpts_scl_ys)) / 2

            A_kpts_scl_xs /= max(np.nanmax(np.abs(A_kpts_scl_xs)), np.nanmax(np.abs(A_kpts_scl_ys)))
            A_kpts_scl_ys /= max(np.nanmax(np.abs(A_kpts_scl_xs)), np.nanmax(np.abs(A_kpts_scl_ys)))
            B_kpts_scl_xs /= max(np.nanmax(np.abs(B_kpts_scl_xs)), np.nanmax(np.abs(B_kpts_scl_ys)))
            B_kpts_scl_ys /= max(np.nanmax(np.abs(B_kpts_scl_xs)), np.nanmax(np.abs(B_kpts_scl_ys)))

            ball_datas = ball_df_values[hf_start:hf_end+1]
            ball_datas[:, 0] = ball_datas[:, 0] / 640 - 1.0
            ball_datas[:, 1] = ball_datas[:, 1] / 360 - 1.0
            ball_datas = np.nan_to_num(ball_datas, nan=0.0)

            if hit_frame-hl < 0:
                A_kpts_ori_xs = np.concatenate([ np.zeros((abs(hit_frame-hl), 17)), A_kpts_ori_xs], axis=0)
                A_kpts_ori_ys = np.concatenate([ np.zeros((abs(hit_frame-hl), 17)), A_kpts_ori_ys], axis=0)
                B_kpts_ori_xs = np.concatenate([ np.zeros((abs(hit_frame-hl), 17)), B_kpts_ori_xs], axis=0)
                B_kpts_ori_ys = np.concatenate([ np.zeros((abs(hit_frame-hl), 17)), B_kpts_ori_ys], axis=0)
                A_kpts_scl_xs = np.concatenate([ np.zeros((abs(hit_frame-hl), 17)), A_kpts_scl_xs], axis=0)
                A_kpts_scl_ys = np.concatenate([ np.zeros((abs(hit_frame-hl), 17)), A_kpts_scl_ys], axis=0)
                B_kpts_scl_xs = np.concatenate([ np.zeros((abs(hit_frame-hl), 17)), B_kpts_scl_xs], axis=0)
                B_kpts_scl_ys = np.concatenate([ np.zeros((abs(hit_frame-hl), 17)), B_kpts_scl_ys], axis=0)
                ball_datas    = np.concatenate([ np.zeros((abs(hit_frame-hl), 2)),  ball_datas   ], axis=0)
            if hit_frame+hl > video_frame_count-1:
                A_kpts_ori_xs = np.concatenate([ A_kpts_ori_xs, np.zeros((hit_frame+hl-(video_frame_count-1), 17)) ], axis=0)
                A_kpts_ori_ys = np.concatenate([ A_kpts_ori_ys, np.zeros((hit_frame+hl-(video_frame_count-1), 17)) ], axis=0)
                B_kpts_ori_xs = np.concatenate([ B_kpts_ori_xs, np.zeros((hit_frame+hl-(video_frame_count-1), 17)) ], axis=0)
                B_kpts_ori_ys = np.concatenate([ B_kpts_ori_ys, np.zeros((hit_frame+hl-(video_frame_count-1), 17)) ], axis=0)
                A_kpts_scl_xs = np.concatenate([ A_kpts_scl_xs, np.zeros((hit_frame+hl-(video_frame_count-1), 17)) ], axis=0)
                A_kpts_scl_ys = np.concatenate([ A_kpts_scl_ys, np.zeros((hit_frame+hl-(video_frame_count-1), 17)) ], axis=0)
                B_kpts_scl_xs = np.concatenate([ B_kpts_scl_xs, np.zeros((hit_frame+hl-(video_frame_count-1), 17)) ], axis=0)
                B_kpts_scl_ys = np.concatenate([ B_kpts_scl_ys, np.zeros((hit_frame+hl-(video_frame_count-1), 17)) ], axis=0)
                ball_datas    = np.concatenate([ ball_datas,    np.zeros((hit_frame+hl-(video_frame_count-1),  2)) ], axis=0)
            
            assert ball_datas.shape == (args.length, 2)

            kpt_imgs = np.zeros((2, args.length, 64, 64))
            for fid, hf in enumerate(range(hit_frame-hl, hit_frame+hl+1)):
                if hf <                  0: continue
                if hf >= video_frame_count: continue
                if pose_df_values[hf, 1] > 0.5:
                    kpts_x = A_kpts_scl_xs[fid]
                    kpts_y = A_kpts_scl_ys[fid]
                    kpts_x = np.array(((kpts_x/2)+0.5) *60 +2, dtype=np.uint8).tolist()
                    kpts_y = np.array(((kpts_y/2)+0.5) *60 +2, dtype=np.uint8).tolist()
                    kpt_imgs[0, fid] = draw_limbs(kpts_x, kpts_y)
                if pose_df_values[hf, 40] > 0.5:
                    kpts_x = B_kpts_scl_xs[fid]
                    kpts_y = B_kpts_scl_ys[fid]
                    kpts_x = np.array(((kpts_x/2)+0.5) *60 +2, dtype=np.uint8).tolist()
                    kpts_y = np.array(((kpts_y/2)+0.5) *60 +2, dtype=np.uint8).tolist()
                    kpt_imgs[1, fid] = draw_limbs(kpts_x, kpts_y)

            kpt_datas = np.concatenate([
                np.expand_dims(np.nan_to_num(A_kpts_ori_xs, nan=0.0), axis=1),
                np.expand_dims(np.nan_to_num(A_kpts_ori_ys, nan=0.0), axis=1),
                np.expand_dims(np.nan_to_num(B_kpts_ori_xs, nan=0.0), axis=1),
                np.expand_dims(np.nan_to_num(B_kpts_ori_ys, nan=0.0), axis=1),
            ], axis=1)
            assert kpt_datas.shape == (args.length, 4, 17)

            time_datas = np.arange(hit_frame-hl, hit_frame+hl+1) / video_frame_count
            assert time_datas.shape == (args.length, )

            bg_id = np.zeros(12)
            if args.mode=="train": bg_id[train_img_to_background[video_id]] = 1
            else                 : bg_id[valid_img_to_background[video_id]] = 1

            hitter = [ 1, 0 ] if hitter=='A' else [ 0, 1 ]
            
            input_imgs.append(kpt_imgs)
            input_kpts.append(kpt_datas)
            input_balls.append(ball_datas)
            input_times.append(time_datas)
            input_bg_ids.append(bg_id)
            input_hitters.append(hitter)

            counter += 1
            if counter == args.batch_size or hid == len(answer_df_values)-1:
                input_imgs    = torch.from_numpy(np.array(input_imgs,  dtype=np.float32)).to(args.device)
                input_kpts    = torch.from_numpy(np.array(input_kpts,  dtype=np.float32)).to(args.device)
                input_balls   = torch.from_numpy(np.array(input_balls, dtype=np.float32)).to(args.device)
                input_times   = torch.from_numpy(np.array(input_times, dtype=np.float32)).to(args.device)
                input_bg_ids  = torch.from_numpy(np.array(input_bg_ids, dtype=np.float32)).to(args.device)
                input_hitters = torch.from_numpy(np.array(input_hitters, dtype=np.float32)).to(args.device)
                pred : torch.Tensor = model(input_imgs, input_kpts, input_balls, input_times, input_bg_ids, input_hitters)
                for p in pred.cpu().detach().numpy(): predictions.append(p)
                counter, input_imgs, input_kpts, input_balls, input_bg_ids, input_times = 0, [], [], [], [], []
        
        predictions = np.array(predictions)
        assert predictions.shape == (len(answer_df_values), 2)

        predictions = np.argmax(predictions, axis=-1)
        assert predictions.shape == (len(answer_df_values), )
        output_df = pd.DataFrame({
            "ShotSeq"  : np.arange(len(answer_df_values)),
            "HitFrame" : answer_df_values[:, 0],
            "Hitter"   : answer_df_values[:, 1],
            "RoundHead": answer_df_values[:, 2],
            "Backhand" : predictions,
        })
        output_df = output_df.set_index("ShotSeq")
        output_df.to_csv(f"data/{args.mode}/{video_id:05}/{video_id:05}_prediction_3_backhand.csv")

    return



""" Execution """
if __name__ == "__main__":

    load_dir = "2023.05.09-16.51.26"
    DEFAULT_MODE       = "valid"
    DEFAULT_BATCH_SIZE = 80
    DEFAULT_DEVICE     = "cuda:0"
    DEFAULT_MODEL_PATH = f"logs/3_backhand/{load_dir}/best_valid_loss.pt"
    # DEFAULT_SAVE_DIR   = f"predictions/3_backhand/{target}"

    parser = argparse.ArgumentParser()
    parser.add_argument("-m",  "--mode",       type=str, default=DEFAULT_MODE)
    parser.add_argument("-bs", "--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("-l",  "--length",     type=int, default=31)
    parser.add_argument("-d",  "--device",     type=str, default=DEFAULT_DEVICE)
    parser.add_argument("-mp", "--model-path", type=str, default=DEFAULT_MODEL_PATH)
    # parser.add_argument("-sd", "--save-dir",   type=str, default=DEFAULT_SAVE_DIR)

    args = parser.parse_args()
    assert args.mode in [ "train", "valid" ]
    main(args)