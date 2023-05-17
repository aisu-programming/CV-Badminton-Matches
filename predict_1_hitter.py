""" Libraries """
import os
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

from dataloader import draw_limbs  # , draw_kpts
from data.background.classification import train_img_to_background, valid_img_to_background, test_img_to_background



""" Functions """
def main(args):

    os.makedirs(args.save_dir, exist_ok=True)

    model = torch.load(args.model_path).to(args.device)
    model = model.eval()

    if   args.mode=="train": video_id_list = list(range(1, 800+1))
    elif args.mode=="valid": video_id_list = list(range(1, 169+1))
    else                   : video_id_list = list(range(170, 399+1))

    for vid, video_id in enumerate(video_id_list):

        if video_id > 100: continue

        ball_df_values = pd.read_csv(f"data/{args.mode}/{video_id:05}/{video_id:05}_ball_33_adj.csv")[["Adjusted X", "Adjusted Y"]].values
        pose_df_values = pd.read_csv(f"data/{args.mode}/{video_id:05}/{video_id:05}_pose_wholebody.csv").values
        video_frame_count = len(pose_df_values)
        length, hl = args.length, (args.length-1)//2
        kpt_count = 133
        predictions, predictions_divider = np.zeros((video_frame_count, 3)), np.zeros((video_frame_count, 1))
        input_kpt_imgs, input_kpts, input_balls, input_bg_ids, input_times = [], [], [], [], []

        for frame in tqdm(range(video_frame_count), desc=f"[{args.mode}] {video_id:05} - Predicting hit frames"):

            hf_start, hf_end = max(frame-hl, 0), min(frame+hl, video_frame_count-1)
            A_kpts_ori_xs = pose_df_values[hf_start:hf_end+1, (  6  ):(  6+kpt_count*2):2] / 640 - 1.0
            A_kpts_ori_ys = pose_df_values[hf_start:hf_end+1, (  6+1):(  6+kpt_count*2):2] / 360 - 1.0
            B_kpts_ori_xs = pose_df_values[hf_start:hf_end+1, (277  ):(277+kpt_count*2):2] / 640 - 1.0
            B_kpts_ori_ys = pose_df_values[hf_start:hf_end+1, (277+1):(277+kpt_count*2):2] / 360 - 1.0
            
            A_kpts_scl_xs = pose_df_values[hf_start:hf_end+1, (  6  ):(  6+kpt_count*2):2]
            A_kpts_scl_ys = pose_df_values[hf_start:hf_end+1, (  6+1):(  6+kpt_count*2):2]
            B_kpts_scl_xs = pose_df_values[hf_start:hf_end+1, (277  ):(277+kpt_count*2):2]
            B_kpts_scl_ys = pose_df_values[hf_start:hf_end+1, (277+1):(277+kpt_count*2):2]

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

            if frame-hl < 0:
                kpt_filler  = np.zeros((abs(frame-hl), kpt_count))
                ball_filler = np.zeros((abs(frame-hl),         2))
                A_kpts_ori_xs = np.concatenate([ kpt_filler,  A_kpts_ori_xs ], axis=0)
                A_kpts_ori_ys = np.concatenate([ kpt_filler,  A_kpts_ori_ys ], axis=0)
                B_kpts_ori_xs = np.concatenate([ kpt_filler,  B_kpts_ori_xs ], axis=0)
                B_kpts_ori_ys = np.concatenate([ kpt_filler,  B_kpts_ori_ys ], axis=0)
                A_kpts_scl_xs = np.concatenate([ kpt_filler,  A_kpts_scl_xs ], axis=0)
                A_kpts_scl_ys = np.concatenate([ kpt_filler,  A_kpts_scl_ys ], axis=0)
                B_kpts_scl_xs = np.concatenate([ kpt_filler,  B_kpts_scl_xs ], axis=0)
                B_kpts_scl_ys = np.concatenate([ kpt_filler,  B_kpts_scl_ys ], axis=0)
                ball_datas    = np.concatenate([ ball_filler, ball_datas    ], axis=0)
            if frame+hl > video_frame_count-1:
                kpt_filler  = np.zeros((frame+hl-(video_frame_count-1), kpt_count))
                ball_filler = np.zeros((frame+hl-(video_frame_count-1),         2))
                A_kpts_ori_xs = np.concatenate([ A_kpts_ori_xs, kpt_filler  ], axis=0)
                A_kpts_ori_ys = np.concatenate([ A_kpts_ori_ys, kpt_filler  ], axis=0)
                B_kpts_ori_xs = np.concatenate([ B_kpts_ori_xs, kpt_filler  ], axis=0)
                B_kpts_ori_ys = np.concatenate([ B_kpts_ori_ys, kpt_filler  ], axis=0)
                A_kpts_scl_xs = np.concatenate([ A_kpts_scl_xs, kpt_filler  ], axis=0)
                A_kpts_scl_ys = np.concatenate([ A_kpts_scl_ys, kpt_filler  ], axis=0)
                B_kpts_scl_xs = np.concatenate([ B_kpts_scl_xs, kpt_filler  ], axis=0)
                B_kpts_scl_ys = np.concatenate([ B_kpts_scl_ys, kpt_filler  ], axis=0)
                ball_datas    = np.concatenate([ ball_datas,    ball_filler ], axis=0)
            assert ball_datas.shape == (length, 2)

            kpt_imgs = np.zeros((2, length, 64, 64))
            for fid, hf in enumerate(range(frame-hl, frame+hl+1)):
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
            assert kpt_datas.shape == (length, 4, kpt_count)

            time_datas = np.arange(frame-hl, frame+hl+1) / video_frame_count

            bg_id = np.zeros(13)
            if   args.mode=="train": bg_id[train_img_to_background[video_id]] = 1
            elif args.mode=="valid": bg_id[valid_img_to_background[video_id]] = 1
            else:
                if test_img_to_background[video_id] == 13:
                    bg_id[11] = 1
                else:
                    bg_id[test_img_to_background[video_id]] = 1
            
            input_kpt_imgs.append(kpt_imgs)
            input_kpts.append(kpt_datas)
            input_balls.append(ball_datas)
            input_times.append(time_datas)
            input_bg_ids.append(bg_id)

            if (frame+1)%args.batch_size==0 or frame==video_frame_count-1:
                input_kpt_imgs = torch.from_numpy(np.array(input_kpt_imgs,  dtype=np.float32)).to(args.device)
                input_kpts     = torch.from_numpy(np.array(input_kpts,  dtype=np.float32)).to(args.device)
                input_balls    = torch.from_numpy(np.array(input_balls, dtype=np.float32)).to(args.device)
                input_times    = torch.from_numpy(np.array(input_times, dtype=np.float32)).to(args.device)
                input_bg_ids   = torch.from_numpy(np.array(input_bg_ids, dtype=np.float32)).to(args.device)
                batch_predictions : torch.Tensor = model(input_kpt_imgs, input_kpts, input_balls, input_times, input_bg_ids)
                pred_frame = (frame+1)-len(batch_predictions)
                for pred in batch_predictions.cpu().detach().numpy():
                    idx_start, idx_end = max(pred_frame-hl, 0),   min(pred_frame+hl, video_frame_count-1)
                    p_start,   p_end   = idx_start-pred_frame+hl, idx_end-pred_frame+hl
                    predictions[idx_start:idx_end+1] += pred[p_start:p_end+1]
                    predictions_divider[idx_start:idx_end+1] += 1
                    pred_frame += 1
                input_kpt_imgs, input_kpts, input_balls, input_bg_ids, input_times = [], [], [], [], []
        
        predictions /= predictions_divider
        assert predictions.shape == (video_frame_count, 3)



        A_hit_prob = np.concatenate([np.ones(5)*0.2, predictions[:, 1], np.ones(5)*0.2], axis=-1)
        A_hit_prob = gaussian_filter1d(A_hit_prob, sigma=1)
        B_hit_prob = np.concatenate([np.ones(5)*0.2, predictions[:, 2], np.ones(5)*0.2], axis=-1)
        B_hit_prob = gaussian_filter1d(B_hit_prob, sigma=1)
        A_peaks, _  = find_peaks(A_hit_prob, distance=16, prominence=0.25)
        B_peaks, _  = find_peaks(B_hit_prob, distance=16, prominence=0.25)
        A_peaks -= 5
        B_peaks -= 5
        A_hit_prob = A_hit_prob[5:-5]
        B_hit_prob = B_hit_prob[5:-5]
        while True:
            break_flag = True
            for pid in range(-1, len(A_peaks)):
                current_peak = A_peaks[pid]   if pid!=-1             else 0
                next_peak    = A_peaks[pid+1] if pid!=len(A_peaks)-1 else 10000
                B_peaks_between = list(filter(lambda bp: current_peak<bp<next_peak, B_peaks.tolist()))
                if len(B_peaks_between) > 1:
                    B_peaks_to_remove = []
                    for bp in B_peaks_between:
                        if bp - current_peak < 8:
                            B_peaks_to_remove.append(bp)
                            B_peaks_between.remove(bp)
                        elif next_peak - bp < 8:
                            B_peaks_to_remove.append(bp)
                            B_peaks_between.remove(bp)
                    if len(B_peaks_between) == 0: raise Exception
                    B_max_peak = sorted(B_peaks_between, key=lambda pid: B_hit_prob[pid], reverse=True)[0]
                    B_peaks_between.remove(B_max_peak)
                    B_peaks_to_remove += B_peaks_between
                    B_peaks = B_peaks.tolist()
                    for bptr in B_peaks_to_remove: B_peaks.remove(bptr)
                    B_peaks = np.array(B_peaks, dtype=int)
                    break_flag = False
            for pid in range(-1, len(B_peaks)):
                current_peak = B_peaks[pid]   if pid!=-1             else 0
                next_peak    = B_peaks[pid+1] if pid!=len(B_peaks)-1 else 10000
                A_peaks_between = list(filter(lambda bp: current_peak<bp<next_peak, A_peaks.tolist()))
                if len(A_peaks_between) > 1:
                    A_peaks_to_remove = []
                    for bp in A_peaks_between:
                        if bp - current_peak < 8:
                            A_peaks_to_remove.append(bp)
                            A_peaks_between.remove(bp)
                        elif next_peak - bp < 8:
                            A_peaks_to_remove.append(bp)
                            A_peaks_between.remove(bp)
                    if len(A_peaks_between) == 0: raise Exception
                    A_max_peak = sorted(A_peaks_between, key=lambda pid: A_hit_prob[pid], reverse=True)[0]
                    A_peaks_between.remove(A_max_peak)
                    A_peaks_to_remove += A_peaks_between
                    A_peaks = A_peaks.tolist()
                    for bptr in A_peaks_to_remove: A_peaks.remove(bptr)
                    A_peaks = np.array(A_peaks, dtype=int)
                    break_flag = False
            if break_flag: break

        
        if args.plot:
            fig = plt.figure(figsize=(20, 5))
            if args.mode == "train":
                hit_df_values = pd.read_csv(f"data/{args.mode}/{video_id:05}/{video_id:05}_S2.csv")[["HitFrame", "Hitter"]].values

            ax = fig.add_subplot(2, 1, 1)
            # ax.plot(np.arange(0, video_frame_count), predictions[:, 0], "o-", c='b', ms=2, lw=1, label="X")
            ax.plot(np.arange(0, video_frame_count), predictions[:, 1], "o-", c='r', ms=2, lw=1, label="A")
            ax.plot(np.arange(0, video_frame_count), predictions[:, 2], "o-", c='g', ms=2, lw=1, label="B")
            ax.legend()
            ax.grid(axis='y')
            ax.set_xticks(np.arange(0, 1500, 50))
            if args.mode == "train":
                for hit_frame, hitter in hit_df_values:
                    if   hitter == 'A': ax.axvline(x=hit_frame, c='r', ls="--", lw=1)
                    elif hitter == 'B': ax.axvline(x=hit_frame, c='g', ls="--", lw=1)
            ax.set_xlim(-10, video_frame_count+10)
            
            ax = fig.add_subplot(2, 1, 2)
            # ax.plot(np.arange(0, video_frame_count), predictions[:, 0], "o-", c='b', ms=2, lw=1, label="X")
            ax.scatter(np.arange(0, video_frame_count)[A_peaks], A_hit_prob[A_peaks], c="black", s=30, label="A peak")
            ax.plot(np.arange(0, video_frame_count), A_hit_prob, "o-", c='r', ms=2, lw=1, label="A")
            ax.scatter(np.arange(0, video_frame_count)[B_peaks], B_hit_prob[B_peaks], c="black", s=30, label="B peak")
            ax.plot(np.arange(0, video_frame_count), B_hit_prob, "o-", c='g', ms=2, lw=1, label="B")
            ax.legend()
            ax.grid(axis='y')
            ax.set_xticks(np.arange(0, 1500, 50))
            if args.mode == "train":
                for hit_frame, hitter in hit_df_values:
                    if   hitter == 'A': ax.axvline(x=hit_frame, c='r', ls="--", lw=1)
                    elif hitter == 'B': ax.axvline(x=hit_frame, c='g', ls="--", lw=1)
            ax.set_xlim(-10, video_frame_count+10)

            if args.mode == "train":
                title = f"{video_id:05}.mp4 - HitFrame count: {len(hit_df_values)} / Peak count: {len(A_peaks)+len(B_peaks)}"
            else:
                title = f"{video_id:05}.mp4 - Peak count: {len(A_peaks)+len(B_peaks)}"

            plt.suptitle(title, fontsize=20)
            plt.title(f"model: {args.model_path}", fontsize=10)
            plt.tight_layout()
            plt.savefig(f"data/{args.mode}/{video_id:05}/{video_id:05}_prediction_1_hitter.png")
            plt.savefig(f"{args.save_dir}/{video_id:05}.png")
            # plt.show()
            plt.close()

        if   len(A_peaks)-len(B_peaks)==1: hitters = ['A','B']*50
        elif len(B_peaks)-len(A_peaks)==1: hitters = ['B','A']*50
        elif len(A_peaks) == len(B_peaks): hitters = ['A','B']*50 if (A_peaks[0]<B_peaks[0]) else ['B','A']*50
        else                             : raise Exception

        hitters = hitters[:int(len(A_peaks)+len(B_peaks))]
        hit_frames = sorted(A_peaks.tolist() + B_peaks.tolist())
        output_df = pd.DataFrame({
            "ShotSeq" : np.arange(len(hitters))+1,
            "HitFrame": np.array(hit_frames),
            "Hitter"  : np.array(hitters),
        })
        output_df = output_df.set_index("ShotSeq")
        output_df.to_csv(f"data/{args.mode}/{video_id:05}/{video_id:05}_prediction_1_hitter.csv")

    return



""" Execution """
if __name__ == "__main__":

    LOAD_DIR           = "2023.05.15-13.55.43"  # HitFrame count:  / HitFrame: 
    DEFAULT_MODE       = "train"
    DEFAULT_DEVICE     = "cuda:1"
    DEFAULT_MODEL_PATH = f"logs/all/1_hitter/{LOAD_DIR}/best_valid_loss.pt"
    DEFAULT_BATCH_SIZE = 90
    DEFAULT_LENGTH     = 45  # 24G Maximum: batch_size * length = 9000
    DEFAULT_PLOT       = True

    parser = argparse.ArgumentParser()
    parser.add_argument("-m",  "--mode",       type=str,  default=DEFAULT_MODE)
    parser.add_argument("-d",  "--device",     type=str,  default=DEFAULT_DEVICE)
    parser.add_argument("-mp", "--model-path", type=str,  default=DEFAULT_MODEL_PATH)
    parser.add_argument("-bs", "--batch-size", type=int,  default=DEFAULT_BATCH_SIZE)
    parser.add_argument("-l",  "--length",     type=int,  default=DEFAULT_LENGTH)
    parser.add_argument("-p",  "--plot",       type=bool, default=DEFAULT_PLOT)

    args = parser.parse_args()
    assert args.mode in [ "train", "valid", "test" ]
    args.save_dir = f"predictions/1_hitter/{LOAD_DIR}_{args.mode}"
    main(args)