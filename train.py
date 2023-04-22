""" Libraries """
import os
import time
import torch
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from typing import Any, Iterator, Tuple
from torch.utils.data import DataLoader

from dataloader import MyMapDataset, split_datasets
from models import MyModel



""" Functions """
class Metric():
    def __init__(self, length) -> None:
        super().__init__()
        self.length = length
        self.losses = []

    def append(self, value) -> None:
        self.losses.append(value)
        if len(self.losses) > self.length: self.losses.pop(0)

    def average(self) -> float:
        return np.average(self.losses)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def train_step(model, batch_input, device):
    batch_videos, batch_datas = batch_input
    batch_videos, batch_datas = batch_videos.to(device), batch_datas.to(device)
    batch_pred = model(batch_videos, batch_datas)
    return batch_pred


def valid_step(model, batch_input, device):
    batch_videos, batch_datas = batch_input
    batch_videos, batch_datas = batch_videos.to(device), batch_datas.to(device)
    batch_pred = model(batch_videos, batch_datas)
    return batch_pred


def train_epoch(model, dataloader, criterion, optimizer, lr_scheduler, device):
    loss_metric, acc_metric = Metric(50), Metric(50)
    pbar = tqdm(enumerate(dataloader), desc="[TRAIN]", total=len(dataloader))
    for batch_i, (batch_input, batch_truth) in pbar:
        batch_pred = train_step(model, batch_input, device)
        batch_truth = batch_truth.to(device)
        loss = criterion(batch_pred, batch_truth)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        loss_metric.append(loss.item())
        batch_truth = batch_truth.cpu().detach().numpy()
        batch_pred  = batch_pred.cpu().detach().numpy()
        acc = np.average(np.argmax(batch_pred, axis=1)==batch_truth)
        acc_metric.append(acc)
        pbar.set_description(f"[TRAIN] loss: {loss_metric.average():.5f}, Acc: {acc_metric.average()*100:.2f}%, LR: {get_lr(optimizer):.10f}")
        
        if (batch_i+1) % 100 == 0:
            with np.printoptions(formatter={'float': '{:5.03f}'.format}):
                print("\nbatch_truth:", batch_truth[0])
                print(  "batch_pred :", np.argmax(batch_pred[0], axis=0))
                print(  "batch_pred :", batch_pred[0])
    return


def valid_epoch(model, dataloader, criterion, device):
    loss_metric, acc_metric = Metric(50), Metric(50)
    pbar = tqdm(enumerate(dataloader), desc="[VALID]", total=len(dataloader))
    for batch_i, (batch_input, batch_truth) in pbar:

        batch_pred = valid_step(model, batch_input, device)
        batch_truth = batch_truth.to(device)
        loss = criterion(batch_pred, batch_truth)

        loss_metric.append(loss.item())
        batch_truth = batch_truth.cpu().detach().numpy()
        batch_pred  = batch_pred.cpu().detach().numpy()
        acc = np.average(np.argmax(batch_pred, axis=1)==batch_truth)
        acc_metric.append(acc)
        pbar.set_description(f"[VALID] loss: {loss_metric.average():.5f}, Acc: {acc_metric.average()*100:.2f}%")

        if (batch_i+1) % 100 == 0:
            with np.printoptions(formatter={'float': '{:5.03f}'.format}):
                print("\nbatch_truth:", batch_truth[0])
                print(  "batch_pred :", np.argmax(batch_pred[0], axis=0))
                print(  "batch_pred :", batch_pred[0])
    return


def main(args):
    # os.makedirs(args.save_dir, exist_ok=True)
    # shutil.copy(__file__, args.save_dir)
    # shutil.copy("models.py", args.save_dir)
    # shutil.copy("dataloader.py", args.save_dir)

    my_dataset = MyMapDataset(args.length, step=args.step)
    # my_dataset = MyMapDataset(args.length, step=args.length)

    # (images_0, datas_0), truths_0 = my_dataset[0]
    # (images_1, datas_1), truths_1 = my_dataset[1]
    # images = torch.concat([images_0[None, ...], images_1[None, ...]], dim=0)
    # datas  = torch.concat([datas_0[None, ...],  datas_1[None, ...]],  dim=0)
    # truths = torch.concat([truths_0[None, ...], truths_1[None, ...]], dim=0)
    # print(images.shape, datas.shape, truths.shape)
    # # return

    my_train_dataset, my_valid_dataset = split_datasets(my_dataset)
    my_train_dataLoader = DataLoader(
        my_train_dataset, args.batch_size, shuffle=True,
        pin_memory=True, drop_last=False,
        num_workers=args.num_workers, collate_fn=None,
        # timeout=0, worker_init_fn=None, sampler=None, batch_sampler=None,
        # multiprocessing_context=None, generator=None,
        # prefetch_factor=2, persistent_workers=False
    )
    my_valid_dataLoader = DataLoader(
        my_valid_dataset, args.batch_size, shuffle=True,
        pin_memory=True, drop_last=False,
        num_workers=args.num_workers, collate_fn=None,
        # timeout=0, worker_init_fn=None, sampler=None, batch_sampler=None,
        # multiprocessing_context=None, generator=None,
        # prefetch_factor=2, persistent_workers=False
    )

    model        = MyModel(args.length).to(args.device)
    criterion    = torch.nn.CrossEntropyLoss()
    optimizer    = torch.optim.Adam(model.parameters(), lr=0.003)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9975)

    for epoch in range(1, args.epochs+1):
        print(f"{epoch}/{args.epochs}")
        train_epoch(model, my_train_dataLoader, criterion, optimizer, lr_scheduler, args.device)
        valid_epoch(model, my_valid_dataLoader, criterion, args.device)
    return



""" Execution """
if __name__ == "__main__":

    DEFAULT_DEVICE   = "cuda:0"
    DEFAULT_SAVE_DIR = f"logs/{time.strftime('%Y.%m.%d-%H.%M.%S', time.localtime())}"

    parser = argparse.ArgumentParser()
    parser.add_argument("-e",  "--epochs",      type=int, default=100)
    parser.add_argument("-bs", "--batch-size",  type=int, default=4)
    parser.add_argument("-nw", "--num-workers", type=int, default=2)
    parser.add_argument("-l",  "--length",      type=int, default=30)
    parser.add_argument("-s",  "--step",        type=int, default=30)
    parser.add_argument("-d",  "--device",      type=str, default=DEFAULT_DEVICE)
    parser.add_argument("-sd", "--save-dir",    type=str, default=DEFAULT_SAVE_DIR)

    args = parser.parse_args()
    main(args)