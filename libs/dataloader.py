""" Libraries """
import cv2
# import mmcv
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Any
from torch.utils.data import Dataset

from data.background.classification import train_img_to_background



""" Functions """
def draw_kpts(kpts_x, kpts_y, important_kpt) -> np.ndarray:
    assert important_kpt < len(kpts_x)
    img = np.zeros((64, 64))
    for coord_id, (x, y) in enumerate(zip(kpts_x, kpts_y)):
        if coord_id == important_kpt: continue
        img = cv2.circle(img, (x, y), radius=0, color=0.3, thickness=-1)
    ikpt = important_kpt
    img = cv2.circle(img, (kpts_x[ikpt], kpts_y[ikpt]), radius=0, color=0.9, thickness=-1)
    return img
    

def draw_limbs(kpts_x, kpts_y) -> np.ndarray:
    assert len(kpts_x)==17 and len(kpts_y)==17
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
    # return (np.array(img)-0.5)*2



""" Classes """
class HitterDataset(Dataset):
    def __init__(self, video_id_list, sample, length) -> None:
        self.video_id_list = video_id_list
        self.sample        = sample
        self.length        = length
        self.load_dataset()

    def load_dataset(self) -> None:
        hl = (self.length-1) // 2  # half_length
        self.ball_type_count = [0]*3
        kpt_count = 23
        input_imgs, input_kpts, input_balls, input_times, input_bg_ids, ground_truths = [], [], [], [], [], []

        for video_id in tqdm(self.video_id_list, desc="Preparing HitterDataset"):

            hit_df_values  = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_S2.csv")[["HitFrame", "Hitter", "BallType"]].values
            pose_df_values = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_pose_wholebody.csv").values
            ball_df_values = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_ball_33_adj.csv")[["Adjusted X", "Adjusted Y"]].values
            video_frame_count = len(pose_df_values)

            random_frame_list = list(range(video_frame_count))
            random.shuffle(random_frame_list)
            random_frame_list = random_frame_list[:self.sample]
            for frame in random_frame_list:
                
                ball_types = np.zeros((self.length), dtype=np.uint8)
                for hf, htr, _ in hit_df_values:
                    if abs(frame-hf) <= hl+1:
                        if   htr == 'A': ball_type = 1
                        elif htr == 'B': ball_type = 2
                        # hf_start, hf_end = max(0, hl-(frame-hf)-1), min(hl-(frame-hf)+1, self.length-1)
                        hf_start, hf_end = max(0, hl-(frame-hf)), min(hl-(frame-hf), self.length-1)
                        ball_types[hf_start:hf_end+1] = ball_type

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
                assert ball_datas.shape == (self.length, 2)

                # import os
                # os.makedirs(f"outputs/{video_id:05}_{frame}_A", exist_ok=True)
                # os.makedirs(f"outputs/{video_id:05}_{frame}_B", exist_ok=True)

                # kpt_imgs = np.ones((2, self.length, 64, 64)) *0.5
                kpt_imgs = np.zeros((2, self.length, 64, 64))
                for fid, _f in enumerate(range(frame-hl, frame+hl+1)):
                    if _f <                  0: continue
                    if _f >= video_frame_count: continue
                    if pose_df_values[_f, 1] > 0.5:
                        kpts_x = A_kpts_scl_xs[fid, :17]
                        kpts_y = A_kpts_scl_ys[fid, :17]
                        kpts_x = np.array(((kpts_x/2)+0.5) *60 +2, dtype=np.uint8).tolist()
                        kpts_y = np.array(((kpts_y/2)+0.5) *60 +2, dtype=np.uint8).tolist()
                        kpt_imgs[0, fid] = draw_limbs(kpts_x, kpts_y)
                        # cv2.imwrite(f"outputs/{video_id:05}_{frame}_A/{fid:03}.jpg", kpt_imgs[0, fid]*255)
                        # cv2.imshow(f"{video_id:05}_{frame}_A", cv2.resize(kpt_imgs[0, fid], (512, 512)))
                        # cv2.waitKey(2)
                        # cv2.destroyAllWindows()
                        # for kpt_id in range(17): kpt_imgs[kpt_id+1, fid] = draw_kpts(kpts_x, kpts_y, kpt_id)
                    if pose_df_values[_f, 272] > 0.5:
                        kpts_x = B_kpts_scl_xs[fid, :17]
                        kpts_y = B_kpts_scl_ys[fid, :17]
                        kpts_x = np.array(((kpts_x/2)+0.5) *60 +2, dtype=np.uint8).tolist()
                        kpts_y = np.array(((kpts_y/2)+0.5) *60 +2, dtype=np.uint8).tolist()
                        kpt_imgs[1, fid] = draw_limbs(kpts_x, kpts_y)
                        # cv2.imwrite(f"outputs/{video_id:05}_{frame}_B/{fid:03}.jpg", kpt_imgs[1, fid]*255)
                        # cv2.imshow(f"{video_id:05}_{frame}_B", cv2.resize(kpt_imgs[1, fid], (512, 512)))
                        # cv2.waitKey(2)
                        # cv2.destroyAllWindows()
                        # for kpt_id in range(17): kpt_imgs[kpt_id+18+1, fid] = draw_kpts(kpts_x, kpts_y, kpt_id)

                # raise Exception
                kpt_datas = np.concatenate([
                    np.expand_dims(np.nan_to_num(A_kpts_ori_xs, nan=0.0), axis=1),
                    np.expand_dims(np.nan_to_num(A_kpts_ori_ys, nan=0.0), axis=1),
                    np.expand_dims(np.nan_to_num(B_kpts_ori_xs, nan=0.0), axis=1),
                    np.expand_dims(np.nan_to_num(B_kpts_ori_ys, nan=0.0), axis=1),
                ], axis=1)
                assert kpt_datas.shape == (self.length, 4, kpt_count)

                time_datas = np.arange(frame-hl, frame+hl+1) / video_frame_count
                assert time_datas.shape == (self.length, )

                bg_id = np.zeros(13)
                bg_id[train_img_to_background[video_id]] = 1

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

    def __getitem__(self, index) -> Any:
        return ((
                self.input_imgs[index],    # ( 2, length,    64, 64)
                self.input_kpts[index],    # ( 4, length, kpt_count)
                self.input_balls[index],   # (    length,         2)
                self.input_times[index],   # (    length           )
                self.input_bg_ids[index],  # (                   12)
            ),
            self.ground_truths[index]      # (                    1)
        )

    def __len__(self) -> int:
        return int(len(self.ground_truths))


class ClassificationDataset(Dataset):
    def __init__(self, length, video_id_list, target) -> None:

        assert target in [ "RoundHead", "Backhand", "BallHeight", "BallType" ]
        hl = (length-1) // 2  # half_length
        self.ground_truth_count = [0]*2 if target!="BallType" else [0]*9
        kpt_count = 23
        input_imgs, input_kpts, input_balls, input_times, input_hitters, input_bg_ids, ground_truths = [], [], [], [], [], [], []

        for video_id in tqdm(video_id_list, desc=f"Preparing ClassificationDataset for {target}"):

            hit_df_values  = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_S2.csv")[["HitFrame", "Hitter", target]].values
            pose_df_values = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_pose_wholebody.csv").values
            ball_df_values = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_ball_33_adj.csv")[["Adjusted X", "Adjusted Y"]].values
            video_frame_count = len(pose_df_values)

            for hit_frame, hitter, ground_truth in hit_df_values:
                
                hf_start, hf_end = max(hit_frame-hl, 0), min(hit_frame+hl, video_frame_count-1)
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

                if hit_frame-hl < 0:
                    kpt_filler  = np.zeros((abs(hit_frame-hl), kpt_count))
                    ball_filler = np.zeros((abs(hit_frame-hl),         2))
                    A_kpts_ori_xs = np.concatenate([ kpt_filler,  A_kpts_ori_xs ], axis=0)
                    A_kpts_ori_ys = np.concatenate([ kpt_filler,  A_kpts_ori_ys ], axis=0)
                    B_kpts_ori_xs = np.concatenate([ kpt_filler,  B_kpts_ori_xs ], axis=0)
                    B_kpts_ori_ys = np.concatenate([ kpt_filler,  B_kpts_ori_ys ], axis=0)
                    A_kpts_scl_xs = np.concatenate([ kpt_filler,  A_kpts_scl_xs ], axis=0)
                    A_kpts_scl_ys = np.concatenate([ kpt_filler,  A_kpts_scl_ys ], axis=0)
                    B_kpts_scl_xs = np.concatenate([ kpt_filler,  B_kpts_scl_xs ], axis=0)
                    B_kpts_scl_ys = np.concatenate([ kpt_filler,  B_kpts_scl_ys ], axis=0)
                    ball_datas    = np.concatenate([ ball_filler, ball_datas    ], axis=0)
                if hit_frame+hl > video_frame_count-1:
                    kpt_filler  = np.zeros((hit_frame+hl-(video_frame_count-1), kpt_count))
                    ball_filler = np.zeros((hit_frame+hl-(video_frame_count-1),         2))
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
                for fid, _f in enumerate(range(hit_frame-hl, hit_frame+hl+1)):
                    if _f <                  0: continue
                    if _f >= video_frame_count: continue
                    if pose_df_values[_f, 1] > 0.5:
                        kpts_x = A_kpts_scl_xs[fid, :17]
                        kpts_y = A_kpts_scl_ys[fid, :17]
                        kpts_x = np.array(((kpts_x/2)+0.5) *60 +2, dtype=np.uint8).tolist()
                        kpts_y = np.array(((kpts_y/2)+0.5) *60 +2, dtype=np.uint8).tolist()
                        kpt_imgs[0, fid] = draw_limbs(kpts_x, kpts_y)
                    if pose_df_values[_f, 272] > 0.5:
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

                time_datas = np.arange(hit_frame-hl, hit_frame+hl+1) / video_frame_count
                assert time_datas.shape == (length, )

                bg_id = np.zeros(13)
                bg_id[train_img_to_background[video_id]] = 1

                hitter = [ 1, 0 ] if hitter=='A' else [ 0, 1 ]

                ground_truth -= 1

                input_imgs.append(kpt_imgs)
                input_kpts.append(kpt_datas)
                input_balls.append(ball_datas)
                input_times.append(time_datas)
                input_hitters.append(hitter)
                input_bg_ids.append(bg_id)
                ground_truths.append(ground_truth)
                self.ground_truth_count[ground_truth] += 1

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
        self.input_hitters = torch.from_numpy(np.array(input_hitters, dtype=np.float32))
        del input_hitters
        self.ground_truths = torch.from_numpy(np.array(ground_truths, dtype=np.uint8)).long()
        del ground_truths

    def __getitem__(self, index) -> Any:
        return ((
                self.input_imgs[index],     # ( 2, length,    64, 64)
                self.input_kpts[index],     # ( 4, length, kpt_count)
                self.input_balls[index],    # (    length,         2)
                self.input_times[index],    # (    length           )
                self.input_bg_ids[index],   # (                   12)
                self.input_hitters[index],  # (                    2)
            ),
            self.ground_truths[index]       # (                    1)
        )

    def __len__(self) -> int:
        return int(len(self.ground_truths))


class BallTypeDataset(Dataset):
    def __init__(self, length, video_id_list) -> None:

        hl = (length-1) // 2  # half_length
        self.ground_truth_count = [0]*9
        kpt_count = 23
        input_imgs, input_kpts, input_balls, input_times, input_hitters, \
            input_bg_ids, input_last_ball_types, ground_truths = [], [], [], [], [], [], [], []

        for video_id in tqdm(video_id_list, desc=f"Preparing BallTypeDataset"):

            hit_df_values  = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_S2.csv")[["HitFrame", "Hitter", "BallType"]].values
            pose_df_values = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_pose_wholebody.csv").values
            ball_df_values = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_ball_33_adj.csv")[["Adjusted X", "Adjusted Y"]].values
            video_frame_count = len(pose_df_values)

            for hid, (hit_frame, hitter, ground_truth) in enumerate(hit_df_values):
                
                hf_start, hf_end = max(hit_frame-hl, 0), min(hit_frame+hl, video_frame_count-1)
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

                if hit_frame-hl < 0:
                    kpt_filler  = np.zeros((abs(hit_frame-hl), kpt_count))
                    ball_filler = np.zeros((abs(hit_frame-hl),         2))
                    A_kpts_ori_xs = np.concatenate([ kpt_filler,  A_kpts_ori_xs ], axis=0)
                    A_kpts_ori_ys = np.concatenate([ kpt_filler,  A_kpts_ori_ys ], axis=0)
                    B_kpts_ori_xs = np.concatenate([ kpt_filler,  B_kpts_ori_xs ], axis=0)
                    B_kpts_ori_ys = np.concatenate([ kpt_filler,  B_kpts_ori_ys ], axis=0)
                    A_kpts_scl_xs = np.concatenate([ kpt_filler,  A_kpts_scl_xs ], axis=0)
                    A_kpts_scl_ys = np.concatenate([ kpt_filler,  A_kpts_scl_ys ], axis=0)
                    B_kpts_scl_xs = np.concatenate([ kpt_filler,  B_kpts_scl_xs ], axis=0)
                    B_kpts_scl_ys = np.concatenate([ kpt_filler,  B_kpts_scl_ys ], axis=0)
                    ball_datas    = np.concatenate([ ball_filler, ball_datas    ], axis=0)
                if hit_frame+hl > video_frame_count-1:
                    kpt_filler  = np.zeros((hit_frame+hl-(video_frame_count-1), kpt_count))
                    ball_filler = np.zeros((hit_frame+hl-(video_frame_count-1),         2))
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
                for fid, _f in enumerate(range(hit_frame-hl, hit_frame+hl+1)):
                    if _f <                  0: continue
                    if _f >= video_frame_count: continue
                    if pose_df_values[_f, 1] > 0.5:
                        kpts_x = A_kpts_scl_xs[fid, :17]
                        kpts_y = A_kpts_scl_ys[fid, :17]
                        kpts_x = np.array(((kpts_x/2)+0.5) *60 +2, dtype=np.uint8).tolist()
                        kpts_y = np.array(((kpts_y/2)+0.5) *60 +2, dtype=np.uint8).tolist()
                        kpt_imgs[0, fid] = draw_limbs(kpts_x, kpts_y)
                    if pose_df_values[_f, 272] > 0.5:
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

                time_datas = np.arange(hit_frame-hl, hit_frame+hl+1) / video_frame_count
                assert time_datas.shape == (length, )

                bg_id = np.zeros(13)
                bg_id[train_img_to_background[video_id]] = 1

                hitter = [ 1, 0 ] if hitter=='A' else [ 0, 1 ]

                last_ball_type = [0]*10
                if hid==0: last_ball_type[0] = 1
                else     : last_ball_type[hit_df_values[hid-1][2]] = 1

                ground_truth -= 1

                input_imgs.append(kpt_imgs)
                input_kpts.append(kpt_datas)
                input_balls.append(ball_datas)
                input_times.append(time_datas)
                input_bg_ids.append(bg_id)
                input_hitters.append(hitter)
                input_last_ball_types.append(last_ball_type)
                ground_truths.append(ground_truth)
                self.ground_truth_count[ground_truth] += 1

        self.input_imgs            = torch.from_numpy(np.array(input_imgs, dtype=np.float32))
        del input_imgs
        self.input_kpts            = torch.from_numpy(np.array(input_kpts, dtype=np.float32))
        del input_kpts
        self.input_balls           = torch.from_numpy(np.array(input_balls, dtype=np.float32))
        del input_balls
        self.input_times           = torch.from_numpy(np.array(input_times, dtype=np.float32))
        del input_times
        self.input_bg_ids          = torch.from_numpy(np.array(input_bg_ids, dtype=np.float32))
        del input_bg_ids
        self.input_hitters         = torch.from_numpy(np.array(input_hitters, dtype=np.float32))
        del input_hitters
        self.input_last_ball_types = torch.from_numpy(np.array(input_last_ball_types, dtype=np.float32))
        del input_last_ball_types
        self.ground_truths         = torch.from_numpy(np.array(ground_truths, dtype=np.uint8)).long()
        del ground_truths

    def __getitem__(self, index) -> Any:
        return ((
                self.input_imgs[index],             # ( 2, length,    64, 64)
                self.input_kpts[index],             # ( 4, length, kpt_count)
                self.input_balls[index],            # (    length,         2)
                self.input_times[index],            # (    length           )
                self.input_bg_ids[index],           # (                   12)
                self.input_hitters[index],          # (                    2)
                self.input_last_ball_types[index],  # (                   10)
            ),
            self.ground_truths[index]               # (                    1)
        )

    def __len__(self) -> int:
        return int(len(self.ground_truths))


class LocationDataset(Dataset):
    def __init__(self, length, video_id_list, target) -> None:
        
        assert target in [ "Landing", "HitterLocation", "DefenderLocation" ]
        hl = (length-1) // 2  # half_length
        kpt_count = 23
        input_imgs, input_kpt_imgs, input_kpts, input_balls, input_times, \
            input_hitters, input_bg_ids, ground_truths = [], [], [], [], [], [], [], []
        
        for video_id in tqdm(video_id_list, desc=f"Preparing LocationDataset for {target}"):

            hit_df_values  = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_S2.csv")[["HitFrame", "Hitter", f"{target}X", f"{target}Y"]].values
            pose_df_values = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_pose_wholebody.csv").values
            ball_df_values = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_ball_33_adj.csv")[["Adjusted X", "Adjusted Y"]].values
            video_frame_count = len(pose_df_values)
            # video_reader = mmcv.VideoReader(f"data/train/{video_id:05}/{video_id:05}.mp4")
            # assert video_reader.opened, "Faild to load video file"

            for hit_frame, hitter, ground_truth_x, ground_truth_y in hit_df_values:
                
                hf_start, hf_end = max(hit_frame-hl, 0), min(hit_frame+hl, video_frame_count-1)
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

                if hit_frame-hl < 0:
                    kpt_filler  = np.zeros((abs(hit_frame-hl), kpt_count))
                    ball_filler = np.zeros((abs(hit_frame-hl),         2))
                    A_kpts_ori_xs = np.concatenate([ kpt_filler,  A_kpts_ori_xs ], axis=0)
                    A_kpts_ori_ys = np.concatenate([ kpt_filler,  A_kpts_ori_ys ], axis=0)
                    B_kpts_ori_xs = np.concatenate([ kpt_filler,  B_kpts_ori_xs ], axis=0)
                    B_kpts_ori_ys = np.concatenate([ kpt_filler,  B_kpts_ori_ys ], axis=0)
                    A_kpts_scl_xs = np.concatenate([ kpt_filler,  A_kpts_scl_xs ], axis=0)
                    A_kpts_scl_ys = np.concatenate([ kpt_filler,  A_kpts_scl_ys ], axis=0)
                    B_kpts_scl_xs = np.concatenate([ kpt_filler,  B_kpts_scl_xs ], axis=0)
                    B_kpts_scl_ys = np.concatenate([ kpt_filler,  B_kpts_scl_ys ], axis=0)
                    ball_datas    = np.concatenate([ ball_filler, ball_datas    ], axis=0)
                if hit_frame+hl > video_frame_count-1:
                    kpt_filler  = np.zeros((hit_frame+hl-(video_frame_count-1), kpt_count))
                    ball_filler = np.zeros((hit_frame+hl-(video_frame_count-1),         2))
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

                # ori_imgs = np.zeros((6, length, 64, 64))
                # for ori_imgs_idx, xy_start_idx in zip([0, 3], [ 6, 277 ]):
                #     if np.isnan(pose_df_values[hf_start:hf_end+1, xy_start_idx]).all(): continue
                #     xs = pose_df_values[hf_start:hf_end+1, xy_start_idx  :xy_start_idx+266:2]
                #     ys = pose_df_values[hf_start:hf_end+1, xy_start_idx+1:xy_start_idx+266:2]
                #     xl, xr = np.nanmin(xs), np.nanmax(xs)
                #     yt, yb = np.nanmin(ys), np.nanmax(ys)
                #     xmid, ymid = (xl+xr)/2, (yt+yb)/2
                #     ext = max(xr-xmid, yb-ymid) *1.2
                #     video = np.array(video_reader[hf_start:hf_end+1])
                #     video = video[:, int(ymid-ext):int(ymid+ext+1), int(xmid-ext):int(xmid+ext+1)]
                #     video = [ cv2.resize(img, (64,64), interpolation=cv2.INTER_CUBIC) for img in (video/255.0) ]
                #
                #     # # Debug
                #     # print(xmid, ymid, ext)
                #     # for imgid, img in enumerate(video):
                #     #     cv2.imshow(f"{video_id:05}.mp4 - Frame: {hf_start+imgid}", img)
                #     #     cv2.waitKey(0)
                #     #     cv2.destroyAllWindows()
                #
                #     if hit_frame-hl < 0:
                #         video = [np.zeros((64,64,3))]*abs(hit_frame-hl) + video
                #     if hit_frame+hl > video_frame_count-1:
                #         video = video + [np.zeros((64,64,3))]*(hit_frame+hl-(video_frame_count-1))
                #     video = np.transpose(np.array(video), (3, 0, 1, 2))
                #     assert video.shape == (3, length, 64, 64)
                #     ori_imgs[ori_imgs_idx:ori_imgs_idx+3] = video

                kpt_imgs = np.zeros((2, length, 64, 64))
                for fid, _f in enumerate(range(hit_frame-hl, hit_frame+hl+1)):
                    if _f <                  0: continue
                    if _f >= video_frame_count: continue
                    if pose_df_values[_f, 1] > 0.5:
                        kpts_x = A_kpts_scl_xs[fid, :17]
                        kpts_y = A_kpts_scl_ys[fid, :17]
                        kpts_x = np.array(((kpts_x/2)+0.5) *60 +2, dtype=np.uint8).tolist()
                        kpts_y = np.array(((kpts_y/2)+0.5) *60 +2, dtype=np.uint8).tolist()
                        kpt_imgs[0, fid] = draw_limbs(kpts_x, kpts_y)
                    if pose_df_values[_f, 272] > 0.5:
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

                time_datas = np.arange(hit_frame-hl, hit_frame+hl+1) / video_frame_count
                assert time_datas.shape == (length, )

                bg_id = np.zeros(13)
                bg_id[train_img_to_background[video_id]] = 1

                hitter = [ 1, 0 ] if hitter=='A' else [ 0, 1 ]

                ground_truth_x /= 1280
                ground_truth_y /= 720

                # input_imgs.append(ori_imgs)
                input_kpt_imgs.append(kpt_imgs)
                input_kpts.append(kpt_datas)
                input_balls.append(ball_datas)
                input_times.append(time_datas)
                input_hitters.append(hitter)
                input_bg_ids.append(bg_id)
                ground_truths.append((ground_truth_x, ground_truth_y))

        # self.input_imgs     = torch.from_numpy(np.array(input_imgs, dtype=np.float32))
        # del input_imgs
        self.input_kpt_imgs = torch.from_numpy(np.array(input_kpt_imgs, dtype=np.float32))
        del input_kpt_imgs
        self.input_kpts     = torch.from_numpy(np.array(input_kpts, dtype=np.float32))
        del input_kpts
        self.input_balls    = torch.from_numpy(np.array(input_balls, dtype=np.float32))
        del input_balls
        self.input_times    = torch.from_numpy(np.array(input_times, dtype=np.float32))
        del input_times
        self.input_bg_ids   = torch.from_numpy(np.array(input_bg_ids, dtype=np.float32))
        del input_bg_ids
        self.input_hitters  = torch.from_numpy(np.array(input_hitters, dtype=np.float32))
        del input_hitters
        self.ground_truths  = torch.from_numpy(np.array(ground_truths, dtype=np.float32))
        del ground_truths

    def __getitem__(self, index) -> Any:
        return ((
                # self.input_imgs[index],     # ( 6, length,    64, 64)
                self.input_kpt_imgs[index], # ( 2, length,    64, 64)
                self.input_kpts[index],     # ( 4, length, kpt_count)
                self.input_balls[index],    # (    length,         2)
                self.input_times[index],    # (    length           )
                self.input_bg_ids[index],   # (                   12)
                self.input_hitters[index],  # (                    2)
            ),
            self.ground_truths[index]       # (                    1)
        )

    def __len__(self) -> int:
        return int(len(self.ground_truths))


class WinnerDataset(Dataset):
    def __init__(self, length, video_id_list) -> None:

        ll = 15             # left_length
        rl = (length-1)-ll  # right_length
        self.ground_truth_count = [0]*2
        kpt_count = 23

        input_imgs, input_kpts, input_balls, input_times, input_hitters, input_bg_ids, ground_truths = [], [], [], [], [], [], []
        for video_id in tqdm(video_id_list, desc=f"Preparing WinnerDataset"):

            hit_df_values  = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_S2.csv")[["HitFrame", "Hitter", "Winner"]].values
            pose_df_values = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_pose_wholebody.csv").values
            ball_df_values = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_ball_33_adj.csv")[["Adjusted X", "Adjusted Y"]].values
            video_frame_count = len(pose_df_values)

            hit_frame, hitter, ground_truth = hit_df_values[-1]
            hf_start, hf_end = max(hit_frame-ll, 0), min(hit_frame+rl, video_frame_count-1)
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

            if hit_frame-ll < 0:
                kpt_filler  = np.zeros((abs(hit_frame-ll), kpt_count))
                ball_filler = np.zeros((abs(hit_frame-ll),         2))
                A_kpts_ori_xs = np.concatenate([ kpt_filler,  A_kpts_ori_xs ], axis=0)
                A_kpts_ori_ys = np.concatenate([ kpt_filler,  A_kpts_ori_ys ], axis=0)
                B_kpts_ori_xs = np.concatenate([ kpt_filler,  B_kpts_ori_xs ], axis=0)
                B_kpts_ori_ys = np.concatenate([ kpt_filler,  B_kpts_ori_ys ], axis=0)
                A_kpts_scl_xs = np.concatenate([ kpt_filler,  A_kpts_scl_xs ], axis=0)
                A_kpts_scl_ys = np.concatenate([ kpt_filler,  A_kpts_scl_ys ], axis=0)
                B_kpts_scl_xs = np.concatenate([ kpt_filler,  B_kpts_scl_xs ], axis=0)
                B_kpts_scl_ys = np.concatenate([ kpt_filler,  B_kpts_scl_ys ], axis=0)
                ball_datas    = np.concatenate([ ball_filler, ball_datas    ], axis=0)
            if hit_frame+rl > video_frame_count-1:
                kpt_filler  = np.zeros((hit_frame+rl-(video_frame_count-1), kpt_count))
                ball_filler = np.zeros((hit_frame+rl-(video_frame_count-1),         2))
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
            for fid, _f in enumerate(range(hit_frame-ll, hit_frame+rl+1)):
                if _f <                  0: continue
                if _f >= video_frame_count: continue
                if pose_df_values[_f, 1] > 0.5:
                    kpts_x = A_kpts_scl_xs[fid, :17]
                    kpts_y = A_kpts_scl_ys[fid, :17]
                    kpts_x = np.array(((kpts_x/2)+0.5) *60 +2, dtype=np.uint8).tolist()
                    kpts_y = np.array(((kpts_y/2)+0.5) *60 +2, dtype=np.uint8).tolist()
                    kpt_imgs[0, fid] = draw_limbs(kpts_x, kpts_y)
                if pose_df_values[_f, 272] > 0.5:
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

            time_datas = np.arange(hit_frame-ll, hit_frame+rl+1) / video_frame_count
            assert time_datas.shape == (length, )

            bg_id = np.zeros(14)
            bg_id[train_img_to_background[video_id]] = 1

            hitter = [ 1, 0 ] if hitter=='A' else [ 0, 1 ]

            ground_truth = 0 if ground_truth=='A' else 1

            input_imgs.append(kpt_imgs)
            input_kpts.append(kpt_datas)
            input_balls.append(ball_datas)
            input_times.append(time_datas)
            input_hitters.append(hitter)
            input_bg_ids.append(bg_id)
            ground_truths.append(ground_truth)
            self.ground_truth_count[ground_truth] += 1

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
        self.input_hitters = torch.from_numpy(np.array(input_hitters, dtype=np.float32))
        del input_hitters
        self.ground_truths = torch.from_numpy(np.array(ground_truths, dtype=np.uint8)).long()
        del ground_truths

    def __getitem__(self, index) -> Any:
        return ((
                self.input_imgs[index],     # ( 2, length,    64, 64)
                self.input_kpts[index],     # ( 4, length, kpt_count)
                self.input_balls[index],    # (    length,         2)
                self.input_times[index],    # (    length           )
                self.input_bg_ids[index],   # (                   12)
                self.input_hitters[index],  # (                    2)
            ),
            self.ground_truths[index]       # (                    1)
        )

    def __len__(self) -> int:
        return int(len(self.ground_truths))