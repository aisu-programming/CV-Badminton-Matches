""" Libraries """
import os
import cv2
import torch
import torch.utils.data
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Any, Iterator, Tuple
from torch.utils.data import Dataset, IterableDataset, Subset

from misc import train_formal_list
from data.train_background.classification import img_to_background



""" Classes """
# class MyMapDataset(Dataset):
#     def __init__(self, length, step=1) -> None:
#         videos, backgrounds, video_lengths = [], [], []
#         input_data, ground_truth = [], []
#         for video_id in tqdm(train_formal_list, desc="Preparing MyMapDataset"):

#             if video_id > 100: break

#             vl   = int(cv2.VideoCapture(f"data/train/{video_id:05}/{video_id:05}.mp4").get(cv2.CAP_PROP_FRAME_COUNT))
#             data = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_combined.csv")
#             data = data.drop("Frame", axis=1).values
#             gt   = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_S2_hit.csv")
#             gt   = gt[["Hit"]].values
#             # gt   = gt.drop("Frame", axis=1).values
#             assert vl == len(data) == len(gt)
#             video_lengths.append(vl)
#             input_data.append(np.float32(data))
#             ground_truth.append(np.float32(gt))
#             images = []
#             for img_id in range(vl):
#                 with open(f"data/train/{video_id:05}/images_0.5/{img_id:04}.jpg", "rb") as img_file:
#                     images.append(img_file.read())
#             videos.append(images)

#         for bg_filename in os.listdir("data/train_background"):
#             if ".png" in bg_filename:
#                 bg_img = cv2.imread(f"data/train_background/cropped/{bg_filename}")
#                 # bg_img = cv2.resize(bg_img, (640, 360), interpolation=cv2.INTER_CUBIC)
#                 backgrounds.append(bg_img)

#         self.videos       = videos
#         self.backgrounds  = backgrounds
#         self.input_data   = input_data
#         self.ground_truth = ground_truth
#         self.indexs = []
#         for video_id, vl in enumerate(video_lengths):
#             for frame in range(0, vl-length+1, step):
#                 self.indexs.append((video_id, frame, frame+length, frame+length//2))

#     def __getitem__(self, index: int) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:

#         video_id, idx_start, idx_end, idx_mid = self.indexs[index]
#         datas  = self.input_data[video_id][idx_start:idx_end]
#         truths = self.ground_truth[video_id][idx_mid]

#         # import time
#         bg_id  = img_to_background[train_formal_list[video_id]]
#         images = [ self.backgrounds[bg_id] ]
#         # timer = time.time()
#         for img_id in range(idx_start, idx_end):
#             img = cv2.imdecode(np.frombuffer(self.videos[video_id][img_id], dtype=np.uint8), -1)
#             img = cv2.copyMakeBorder(img, 70, 70, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
#             img = img[:, 70:-70]
#             images.append(img)
#         # print(time.time()-timer)
#         images = np.array(images, dtype=np.float32) / 255
#         images = images.transpose((0, 3, 1, 2))
#         # images = images.reshape((-1, 360, 640))
#         images = images.reshape((-1, 500, 500))

#         images = torch.from_numpy(images)
#         datas  = torch.from_numpy(datas)
#         truths = torch.from_numpy(truths).squeeze()
#         return (images, datas), truths

#     def __len__(self) -> int:
#         return len(self.indexs)
    

class MyMapDataset(Dataset):
    def __init__(self) -> None:

        clip_length = 7
        hit_sets, idle_sets = [], []
        video_images, input_data, ground_truth = [], [], []
        for video_id, train_formal_video_id in tqdm(enumerate(train_formal_list),
                                                    desc="Preparing MyMapDataset",
                                                    total=len(train_formal_list)):
            if video_id > 100: break

            tfvid = train_formal_video_id

            frame_count = int(cv2.VideoCapture(f"data/train/{tfvid:05}/{tfvid:05}.mp4").get(cv2.CAP_PROP_FRAME_COUNT))
            hit_frame   = pd.read_csv(f"data/train/{tfvid:05}/{tfvid:05}_S2.csv")[["HitFrame"]].values.flatten()
            idle_index  = np.array([ True ] * frame_count)
            for hf in hit_frame:
                hit_sets.append((int(video_id), int(hf)))
                idle_index[int(hf-clip_length):int(hf+clip_length)] = False
            idle_index[-clip_length:] = False
            idle_frame = np.arange(frame_count)[idle_index]
            for i_f in idle_frame:
                idle_sets.append((int(video_id), int(i_f)))

            images_dir = f"data/train/{tfvid:05}/images_0.25"
            data = pd.read_csv(f"data/train/{tfvid:05}/{tfvid:05}_combined.csv")
            data = data.drop("Frame", axis=1).values
            gt   = pd.read_csv(f"data/train/{tfvid:05}/{tfvid:05}_S2_hit.csv")
            # gt   = gt.drop("Frame", axis=1).values
            gt   = gt[["Hit"]].values
            assert frame_count == len(data) == len(gt)
            input_data.append(np.float32(data))
            ground_truth.append(np.float32(gt))
            images = []
            for img_filename in os.listdir(images_dir):
                with open(f"{images_dir}/{img_filename}", "rb") as img_file:
                    images.append(img_file.read())
            video_images.append(images)

        backgrounds = []
        for bg_filename in os.listdir("data/train_background"):
            if ".png" in bg_filename:
                bg_img = cv2.imread(f"data/train_background/images_0.25/{bg_filename}")
                # bg_img = cv2.resize(bg_img, (640, 360), interpolation=cv2.INTER_CUBIC)
                backgrounds.append(bg_img)

        self.clip_length  = clip_length
        self.backgrounds  = backgrounds
        self.videos       = video_images
        self.input_data   = input_data
        self.ground_truth = ground_truth
        self.hit_sets     = hit_sets
        self.idle_sets    = idle_sets

    def __getitem__(self, index) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:

        if index % 2 == 1:
            iter_set = self.idle_sets
            index = random.randrange(0, len(iter_set))
        else:
            iter_set = self.hit_sets
            index = index // 2

        video_id, frame = iter_set[index]
        datas  = self.input_data[video_id][frame:frame+self.clip_length]
        truths = self.ground_truth[video_id][frame]

        bg_id  = img_to_background[train_formal_list[video_id]]
        images = [ self.backgrounds[bg_id] ]
        for img_frame in range(frame, frame+self.clip_length):
            img = cv2.imdecode(np.frombuffer(self.videos[video_id][img_frame], dtype=np.uint8), -1)
            # cv2.imshow("", img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            images.append(img)
        images = np.array(images, dtype=np.float32) / 255
        images = images.transpose((0, 3, 1, 2))
        images = images.reshape((-1, 250, 250))

        images = torch.from_numpy(images)
        datas  = torch.from_numpy(datas)
        truths = torch.from_numpy(truths).squeeze()
        return (images, datas), truths

    def __len__(self) -> int:
        return int(len(self.hit_sets) *2)


# class MyIterableDataset(IterableDataset):
#     def __init__(self, start: int, end: int) -> None:
#         super(MyIterableDataset).__init__()
#         assert end > start, "this example code only works with end >= start"
#         self.start = start
#         self.end = end
#
#     def __iter__(self) -> Iterator[int]:
#         worker_info = torch.utils.data.get_worker_info()
#         if worker_info is None:
#             iter_start = self.start
#             iter_end = self.end
#         else:
#             per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
#             worker_id = worker_info.id
#             iter_start = self.start + worker_id * per_worker
#             iter_end = min(iter_start + per_worker, self.end)
#         return iter(range(iter_start, iter_end))



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