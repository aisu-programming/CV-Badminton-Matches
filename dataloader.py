""" Libraries """
import cv2
import math
import torch
import torch.utils.data
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Any, Iterator, Tuple
from torch.utils.data import Dataset, IterableDataset, Subset

from misc import train_formal_list



""" Classes """
class MyMapDataset(Dataset):
    def __init__(self, length, step=1) -> None:
        super(MyMapDataset).__init__()
        
        video_lengths, input_data, ground_truth = [], [], []
        for video_id in tqdm(train_formal_list, desc="Preparing MyMapDataset"):

            if video_id > 300: break

            vl   = int(cv2.VideoCapture(f"data/train/{video_id:05}/{video_id:05}.mp4").get(cv2.CAP_PROP_FRAME_COUNT))
            data = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_combined.csv")
            data = data.drop("Frame", axis=1).values
            gt   = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_S2_hit.csv")
            gt   = gt.drop("Frame", axis=1).values
            assert vl == len(data) == len(gt)
            video_lengths.append(vl)
            input_data.append(np.float32(data))
            ground_truth.append(gt)

        self.input_data   = input_data
        self.ground_truth = ground_truth
        self.indexs = []
        for video_id, vl in enumerate(video_lengths):
            for frame in range(0, vl-length+1, step):
                self.indexs.append((video_id, frame, frame+length))

    def __getitem__(self, index: int) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:

        video_id, idx_start, idx_end = self.indexs[index]
        datas  = self.input_data[video_id][idx_start:idx_end]
        truths = self.ground_truth[video_id][idx_start:idx_end]

        images = []
        dir_vid_id = train_formal_list[video_id]
        for img_id in range(idx_start, idx_end):
            images.append(cv2.imread(f"data/train/{dir_vid_id:05}/images/{img_id:04}.jpg"))
        images = np.array(images, dtype=np.float32) / 255
        images = images.transpose((0, 3, 1, 2))
        # images = images.reshape((720, 1280, -1))

        images = torch.from_numpy(images)
        datas  = torch.from_numpy(datas)
        truths = torch.from_numpy(truths).squeeze()
        return (images, datas), truths

    def __len__(self) -> int:
        return len(self.indexs)


# class MyIterableDataset(IterableDataset):
#     def __init__(self, start: int, end: int) -> None:
#         super(MyIterableDataset).__init__()
#         assert end > start, "this example code only works with end >= start"
#         self.start = start
#         self.end = end
        
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