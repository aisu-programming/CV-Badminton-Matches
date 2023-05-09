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

from dataloader import SELECTED_COLUMNS, MyMapDatasetV4, split_datasets
from models import MyModelV3



""" Functions """
class Metric():
    def __init__(self, length) -> None:
        super().__init__()
        self.length = length
        self.values = []

    def append(self, value) -> None:
        self.values.append(value)
        if len(self.values) > self.length: self.values.pop(0)
        return

    def average(self) -> float:
        return np.average(self.values)


def get_lr(optimizer) -> float:
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def forward_step(model, batch_datas, device) -> torch.Tensor:
    batch_datas = batch_datas.to(device)
    batch_pred = model(batch_datas)
    return batch_pred


def train_epoch(model, dataloader, criterion, optimizer, lr_scheduler, device) -> None:
    loss_metric, acc_metric = Metric(30), Metric(30)
    pbar = tqdm(enumerate(dataloader), desc="[TRAIN]", total=len(dataloader))
    for batch_i, (batch_input, batch_truth) in pbar:
        batch_pred  = forward_step(model, batch_input, device)
        batch_truth = batch_truth.to(device)

        # print(batch_pred.shape, batch_truth.shape)
        loss = criterion(batch_pred, batch_truth)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        loss_metric.append(loss.item())
        batch_truth = batch_truth.cpu().detach().numpy()
        batch_pred  = batch_pred.cpu().detach().numpy()
        acc = np.average(np.round(batch_pred)==batch_truth)
        # acc = np.average(np.sum(batch_pred>0.5, axis=-1)==np.sum(batch_truth>0.5, axis=-1))
        acc_metric.append(acc)
        pbar.set_description(f"[TRAIN] loss: {loss_metric.average():.5f}, Acc: {acc_metric.average()*100:.3f}%, LR: {get_lr(optimizer):.10f}")
        
    with np.printoptions(formatter={'float': '{:6.02f}'.format}):
        print("batch_pred :", batch_pred[0, :10])
        print("batch_pred :", np.round(batch_pred[0, :10]))
        print("batch_truth:", batch_truth[0, :10])
        # print(np.sum(batch_pred>0.5, axis=-1), np.sum(batch_truth>0.5, axis=-1))
        # print(np.sum(batch_pred, axis=-1), np.sum(batch_truth, axis=-1))
    return


def valid_epoch(model, dataloader, criterion, device) -> None:
    losses, accuracies = [], []
    pbar = tqdm(enumerate(dataloader), desc="[VALID]", total=len(dataloader))
    for batch_i, (batch_input, batch_truth) in pbar:

        batch_pred  = forward_step(model, batch_input, device)
        batch_truth = batch_truth.to(device)
        loss = criterion(batch_pred, batch_truth)

        losses.append(loss.item())
        batch_truth = batch_truth.cpu().detach().numpy()
        batch_pred  = batch_pred.cpu().detach().numpy()
        acc = np.average(np.round(batch_pred)==batch_truth)
        # acc = np.average(np.sum(batch_pred>0.5, axis=-1)==np.sum(batch_truth>0.5, axis=-1))
        accuracies.append(acc)
        pbar.set_description(f"[VALID] loss: {np.average(losses):.5f}, Acc: {np.average(accuracies)*100:.3f}%")

    with np.printoptions(formatter={'float': '{:6.02f}'.format}):
        print("batch_pred :", batch_pred[0, :10])
        print("batch_pred :", np.round(batch_pred[0, :10]))
        print("batch_truth:", batch_truth[0, :10])
        # print(np.sum(batch_pred>0.5, axis=-1), np.sum(batch_truth>0.5, axis=-1))
        # print(np.sum(batch_pred, axis=-1), np.sum(batch_truth, axis=-1))
    return


def main(args) -> None:
    # os.makedirs(args.save_dir, exist_ok=True)
    # shutil.copy(__file__, args.save_dir)
    # shutil.copy("models.py", args.save_dir)
    # shutil.copy("dataloader.py", args.save_dir)

    my_dataset = MyMapDatasetV4()
    my_train_dataset, my_valid_dataset = split_datasets(my_dataset)
    my_train_dataLoader = DataLoader(
        my_train_dataset, args.batch_size, shuffle=True,
        pin_memory=True, drop_last=True,
        num_workers=args.num_workers,
        # collate_fn=None, worker_init_fn=None, 
        # timeout=0, sampler=None, batch_sampler=None,
        # multiprocessing_context=None, generator=None,
        # prefetch_factor=2, persistent_workers=False
    )
    my_valid_dataLoader = DataLoader(
        my_valid_dataset, args.batch_size, shuffle=True,
        pin_memory=True, drop_last=True,
        num_workers=args.num_workers,
        # collate_fn=None, worker_init_fn=None, 
        # timeout=0, sampler=None, batch_sampler=None,
        # multiprocessing_context=None, generator=None,
        # prefetch_factor=2, persistent_workers=False
    )

    model        = MyModelV3(len(SELECTED_COLUMNS)).to(args.device)
    criterion    = torch.nn.MSELoss()
    optimizer    = torch.optim.Adam(model.parameters(), lr=0.005)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9975)

    for epoch in range(1, args.epochs+1):
        print(f"\n{epoch}/{args.epochs}")
        train_epoch(model, my_train_dataLoader, criterion, optimizer, lr_scheduler, args.device)
        valid_epoch(model, my_valid_dataLoader, criterion, args.device)
    return



""" Execution """
if __name__ == "__main__":

    DEFAULT_DEVICE   = "cuda:0"
    DEFAULT_SAVE_DIR = f"logs/{time.strftime('%Y.%m.%d-%H.%M.%S', time.localtime())}"

    parser = argparse.ArgumentParser()
    parser.add_argument("-e",  "--epochs",      type=int, default=100)
    parser.add_argument("-bs", "--batch-size",  type=int, default=10)
    parser.add_argument("-nw", "--num-workers", type=int, default=4)
    parser.add_argument("-d",  "--device",      type=str, default=DEFAULT_DEVICE)
    parser.add_argument("-sd", "--save-dir",    type=str, default=DEFAULT_SAVE_DIR)

    args = parser.parse_args()
    main(args)