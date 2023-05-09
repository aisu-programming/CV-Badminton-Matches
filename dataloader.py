""" Libraries """
import cv2
import torch
import torch.utils.data
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Any, Iterator, Tuple, List
from torch.utils.data import Dataset, Subset

from data.train_background.classification import img_to_background as train_img_to_background
from data.valid_background.classification import img_to_background as valid_img_to_background



""" Classes """
def draw_kpts(kpts_x, kpts_y, important_kpt):
    assert important_kpt < len(kpts_x)
    img = np.zeros((64, 64))
    for coord_id, (x, y) in enumerate(zip(kpts_x, kpts_y)):
        if coord_id == important_kpt: continue
        img = cv2.circle(img, (x, y), radius=0, color=0.3, thickness=-1)
    ikpt = important_kpt
    img = cv2.circle(img, (kpts_x[ikpt], kpts_y[ikpt]), radius=0, color=0.9, thickness=-1)
    return img
    

def draw_limbs(kpts_x, kpts_y):
    img = np.ones((64, 64)) * 0.5
    lines = [ (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),  # nose -> others
              (5, 7), (7, 9), (6, 8), (8, 10),  # arms
              (5, 11), (6, 12), (11, 12),  # body
              (11, 13), (13, 15), (12, 14), (14, 16)  # legs
            ]
    for ls, le in lines:
        color = 0.7 if le % 2 == 0 else 0.3
        pt1 = (kpts_x[ls], kpts_y[ls])
        pt2 = (kpts_x[le], kpts_y[le])
        img = cv2.line(img, pt1, pt2, color=color, thickness=1)
    for kpt_id, (x, y) in enumerate(zip(kpts_x, kpts_y)):
        color = 0.9 if kpt_id % 2 == 0 else 0.1
        img = cv2.circle(img, (x, y), radius=0, color=color, thickness=-1)
    return img


class HitterDataset(Dataset):
    def __init__(self, length, video_id_list) -> List[int]:
        
        hl = (length-1) // 2  # half_length
        self.ball_type_count = [ 0 ] * 3

        input_imgs, input_kpts, input_balls, input_times, input_bg_ids, ground_truths = [], [], [], [], [], []
        for video_id in tqdm(video_id_list, desc="Preparing HitterDataset"):

            hit_df_values  = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_S2.csv")[["HitFrame", "Hitter", "BallType"]].values
            pose_df_values = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_pose.csv").values
            ball_df_values = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_ball_33_adj.csv")[["Adjusted X", "Adjusted Y"]].values
            video_frame_count = len(pose_df_values)

            random_frame_list = list(range(video_frame_count))
            random.shuffle(random_frame_list)
            random_frame_list = random_frame_list[:15]
            for frame in random_frame_list:
                
                ball_types = np.zeros((length), dtype=np.uint8)
                for hf, htr, _ in hit_df_values:
                    if abs(frame-hf) <= hl+1:
                        if   htr == 'A': ball_type = 1
                        elif htr == 'B': ball_type = 2
                        # hf_start, hf_end = max(0, hl-(frame-hf)-1), min(hl-(frame-hf)+1, length-1)
                        hf_start, hf_end = max(0, hl-(frame-hf)), min(hl-(frame-hf), length-1)
                        ball_types[hf_start:hf_end+1] = ball_type

                hf_start, hf_end = max(frame-hl, 0), min(frame+hl, video_frame_count-1)
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

                if frame-hl < 0:
                    A_kpts_ori_xs = np.concatenate([ np.zeros((abs(frame-hl), 17)), A_kpts_ori_xs], axis=0)
                    A_kpts_ori_ys = np.concatenate([ np.zeros((abs(frame-hl), 17)), A_kpts_ori_ys], axis=0)
                    B_kpts_ori_xs = np.concatenate([ np.zeros((abs(frame-hl), 17)), B_kpts_ori_xs], axis=0)
                    B_kpts_ori_ys = np.concatenate([ np.zeros((abs(frame-hl), 17)), B_kpts_ori_ys], axis=0)
                    A_kpts_scl_xs = np.concatenate([ np.zeros((abs(frame-hl), 17)), A_kpts_scl_xs], axis=0)
                    A_kpts_scl_ys = np.concatenate([ np.zeros((abs(frame-hl), 17)), A_kpts_scl_ys], axis=0)
                    B_kpts_scl_xs = np.concatenate([ np.zeros((abs(frame-hl), 17)), B_kpts_scl_xs], axis=0)
                    B_kpts_scl_ys = np.concatenate([ np.zeros((abs(frame-hl), 17)), B_kpts_scl_ys], axis=0)
                    ball_datas    = np.concatenate([ np.zeros((abs(frame-hl), 2)),  ball_datas   ], axis=0)
                if frame+hl > video_frame_count-1:
                    A_kpts_ori_xs = np.concatenate([ A_kpts_ori_xs, np.zeros((frame+hl-(video_frame_count-1), 17)) ], axis=0)
                    A_kpts_ori_ys = np.concatenate([ A_kpts_ori_ys, np.zeros((frame+hl-(video_frame_count-1), 17)) ], axis=0)
                    B_kpts_ori_xs = np.concatenate([ B_kpts_ori_xs, np.zeros((frame+hl-(video_frame_count-1), 17)) ], axis=0)
                    B_kpts_ori_ys = np.concatenate([ B_kpts_ori_ys, np.zeros((frame+hl-(video_frame_count-1), 17)) ], axis=0)
                    A_kpts_scl_xs = np.concatenate([ A_kpts_scl_xs, np.zeros((frame+hl-(video_frame_count-1), 17)) ], axis=0)
                    A_kpts_scl_ys = np.concatenate([ A_kpts_scl_ys, np.zeros((frame+hl-(video_frame_count-1), 17)) ], axis=0)
                    B_kpts_scl_xs = np.concatenate([ B_kpts_scl_xs, np.zeros((frame+hl-(video_frame_count-1), 17)) ], axis=0)
                    B_kpts_scl_ys = np.concatenate([ B_kpts_scl_ys, np.zeros((frame+hl-(video_frame_count-1), 17)) ], axis=0)
                    ball_datas    = np.concatenate([ ball_datas,    np.zeros((frame+hl-(video_frame_count-1),  2)) ], axis=0)
                assert ball_datas.shape == (length, 2)

                kpt_imgs = np.zeros((2, length, 64, 64))
                for fid, _f in enumerate(range(frame-hl, frame+hl+1)):
                    if _f <                  0: continue
                    if _f >= video_frame_count: continue
                    if pose_df_values[_f, 1] > 0.5:
                        kpts_x = A_kpts_scl_xs[fid]
                        kpts_y = A_kpts_scl_ys[fid]
                        kpts_x = np.array(((kpts_x/2)+0.5) *60 +2, dtype=np.uint8).tolist()
                        kpts_y = np.array(((kpts_y/2)+0.5) *60 +2, dtype=np.uint8).tolist()
                        kpt_imgs[0, fid] = draw_limbs(kpts_x, kpts_y)
                        # cv2.imshow(f"{video_id:05}_{hit_id}_A", cv2.resize(kpt_imgs[0, fid], (512, 512)))
                        # cv2.waitKey(2)
                        # cv2.destroyAllWindows()
                        # for kpt_id in range(17): kpt_imgs[kpt_id+1, fid] = draw_kpts(kpts_x, kpts_y, kpt_id)
                    if pose_df_values[_f, 40] > 0.5:
                        kpts_x = B_kpts_scl_xs[fid]
                        kpts_y = B_kpts_scl_ys[fid]
                        kpts_x = np.array(((kpts_x/2)+0.5) *60 +2, dtype=np.uint8).tolist()
                        kpts_y = np.array(((kpts_y/2)+0.5) *60 +2, dtype=np.uint8).tolist()
                        kpt_imgs[1, fid] = draw_limbs(kpts_x, kpts_y)
                        # cv2.imshow(f"{video_id:05}_{hit_id}_B", cv2.resize(kpt_imgs[1, fid], (512, 512)))
                        # cv2.waitKey(2)
                        # cv2.destroyAllWindows()
                        # for kpt_id in range(17): kpt_imgs[kpt_id+18+1, fid] = draw_kpts(kpts_x, kpts_y, kpt_id)

                kpt_datas = np.concatenate([  # (4, length, 17)
                    np.expand_dims(np.nan_to_num(A_kpts_ori_xs, nan=0.0), axis=1),
                    np.expand_dims(np.nan_to_num(A_kpts_ori_ys, nan=0.0), axis=1),
                    np.expand_dims(np.nan_to_num(B_kpts_ori_xs, nan=0.0), axis=1),
                    np.expand_dims(np.nan_to_num(B_kpts_ori_ys, nan=0.0), axis=1),
                ], axis=1)
                assert kpt_datas.shape == (length, 4, 17)

                time_datas = np.arange(frame-hl, frame+hl+1) / video_frame_count
                assert time_datas.shape == (length, )

                bg_id = np.zeros(12)
                # if args.mode=="train": bg_id[train_img_to_background[video_id]] = 1
                # else                 : bg_id[valid_img_to_background[video_id]] = 1

                input_imgs.append(kpt_imgs)
                input_kpts.append(kpt_datas)
                input_balls.append(ball_datas)
                input_times.append(time_datas)
                input_bg_ids.append(bg_id)
                ground_truths.append(ball_types)
                for bt in ball_types: self.ball_type_count[bt] += 1

                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

        self.input_imgs    = torch.from_numpy(np.array(input_imgs, dtype=np.float32))
        del input_imgs
        self.input_kpts    = torch.from_numpy(np.array(input_kpts, dtype=np.float32))
        del input_kpts
        self.input_balls   = torch.from_numpy(np.array(input_balls, dtype=np.float32))
        del input_balls
        self.input_times   = torch.from_numpy(np.array(input_times, dtype=np.float32))
        del input_times
        self.input_bg_ids  = torch.from_numpy(np.array(input_bg_ids, dtype=np.float32))
        del input_bg_ids
        self.ground_truths = torch.from_numpy(np.array(ground_truths, dtype=np.uint8)).long()
        del ground_truths

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        return ((
                self.input_imgs[index],    # ( 2, length, 64, 64)
                self.input_kpts[index],    # ( 4, length,     17)
                self.input_balls[index],   # (    length,      2)
                self.input_times[index],   # (    length        )
                self.input_bg_ids[index],  # (                12)
            ),
            self.ground_truths[index]    # (                 1)
        )

    def __len__(self) -> int:
        return int(len(self.ground_truths))
    

class RoundHeadColumnsDataset(Dataset):
    def __init__(self, length, video_id_list) -> List[int]:
        
        hl = (length-1) // 2  # half_length
        self.round_head_count = [ 0 ] * 2

        input_imgs, input_kpts, input_balls, input_times, input_hitters, input_bg_ids, ground_truths = [], [], [], [], [], [], []
        for video_id in tqdm(video_id_list, desc="Preparing RoundHeadColumnsDataset"):

            hit_df_values  = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_S2.csv")[["HitFrame", "Hitter", "RoundHead"]].values
            pose_df_values = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_pose.csv").values
            ball_df_values = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_ball_33_adj.csv")[["Adjusted X", "Adjusted Y"]].values
            video_frame_count = len(pose_df_values)

            for hit_frame, hitter, round_head in hit_df_values:
                
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
                assert ball_datas.shape == (length, 2)

                kpt_imgs = np.zeros((2, length, 64, 64))
                for fid, _f in enumerate(range(hit_frame-hl, hit_frame+hl+1)):
                    if _f <                  0: continue
                    if _f >= video_frame_count: continue
                    if pose_df_values[_f, 1] > 0.5:
                        kpts_x = A_kpts_scl_xs[fid]
                        kpts_y = A_kpts_scl_ys[fid]
                        kpts_x = np.array(((kpts_x/2)+0.5) *60 +2, dtype=np.uint8).tolist()
                        kpts_y = np.array(((kpts_y/2)+0.5) *60 +2, dtype=np.uint8).tolist()
                        kpt_imgs[0, fid] = draw_limbs(kpts_x, kpts_y)
                    if pose_df_values[_f, 40] > 0.5:
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
                assert kpt_datas.shape == (length, 4, 17)

                time_datas = np.arange(hit_frame-hl, hit_frame+hl+1) / video_frame_count
                assert time_datas.shape == (length, )

                bg_id = np.zeros(12)
                # if args.mode=="train": bg_id[train_img_to_background[video_id]] = 1
                # else                 : bg_id[valid_img_to_background[video_id]] = 1

                hitter = [ 1, 0 ] if hitter=='A' else [ 0, 1 ]

                round_head -= 1

                input_imgs.append(kpt_imgs)
                input_kpts.append(kpt_datas)
                input_balls.append(ball_datas)
                input_times.append(time_datas)
                input_hitters.append(hitter)
                input_bg_ids.append(bg_id)
                ground_truths.append(round_head)
                self.round_head_count[round_head] += 1

        self.input_imgs    = torch.from_numpy(np.array(input_imgs, dtype=np.float32))
        del input_imgs
        self.input_kpts    = torch.from_numpy(np.array(input_kpts, dtype=np.float32))
        del input_kpts
        self.input_balls   = torch.from_numpy(np.array(input_balls, dtype=np.float32))
        del input_balls
        self.input_times   = torch.from_numpy(np.array(input_times, dtype=np.float32))
        del input_times
        self.input_bg_ids = torch.from_numpy(np.array(input_bg_ids, dtype=np.float32))
        del input_bg_ids
        self.input_hitters = torch.from_numpy(np.array(input_hitters, dtype=np.float32))
        del input_hitters
        self.ground_truths = torch.from_numpy(np.array(ground_truths, dtype=np.uint8)).long()
        del ground_truths

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        return ((
                self.input_imgs[index],     # ( 2, length, 64, 64)
                self.input_kpts[index],     # ( 4, length,     17)
                self.input_balls[index],    # (    length,      2)
                self.input_times[index],    # (    length        )
                self.input_bg_ids[index],   # (                12)
                self.input_hitters[index],  # (                 2)
            ),
            self.ground_truths[index]    # (                 1)
        )

    def __len__(self) -> int:
        return int(len(self.ground_truths))



""" Functions """
def split_datasets(dataset: Dataset) -> Tuple[Subset, Subset]:
    dataset_length = len(dataset)
    train_dataset_length = int(dataset_length*0.8)
    valid_dataset_length = dataset_length - train_dataset_length
    train_dataset, valid_dataset = \
        torch.utils.data.random_split(
            dataset, [train_dataset_length, valid_dataset_length],
            generator=torch.Generator().manual_seed(0))
    return train_dataset, valid_dataset