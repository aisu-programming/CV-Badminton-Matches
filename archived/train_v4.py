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
from torch.utils.tensorboard import SummaryWriter

from dataloader import MyMapDatasetV5, split_datasets
from models import MyModelV4



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
    loss_metric, acc_metric = Metric(200), Metric(200)
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
        acc = np.average(np.round(batch_pred)==np.round(batch_truth))
        acc_metric.append(acc)
        pbar.set_description(f"[TRAIN] loss: {loss_metric.average():.5f}, Acc: {acc_metric.average()*100:.3f}%, LR: {get_lr(optimizer):.10f}")
        
    with np.printoptions(formatter={'float': '{:5.03f}'.format}):
        print("batch_pred :", batch_pred[:10])
        # print("batch_pred :", np.round(batch_pred[:10]))
        print("batch_truth:", np.round(batch_truth[:10]))
        print("batch_corct:", np.array([" X ", "   "])[np.uint8(np.round(batch_pred[:10])==np.round(batch_truth[:10]))])

    return loss_metric.average(), acc_metric.average()


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
        acc = np.average(np.round(batch_pred)==np.round(batch_truth))
        accuracies.append(acc)
        pbar.set_description(f"[VALID] loss: {np.average(losses):.5f}, Acc: {np.average(accuracies)*100:.3f}%")

    with np.printoptions(formatter={'float': '{:5.03f}'.format}):
        print("batch_pred :", batch_pred[:10])
        # print("batch_pred :", np.round(batch_pred[:10]))
        print("batch_truth:", np.round(batch_truth[:10]))
        print("batch_corct:", np.array([" X ", "   "])[np.uint8(np.round(batch_pred[:10])==np.round(batch_truth[:10]))])
        
    return np.average(losses), np.average(accuracies)


def main(args) -> None:
    os.makedirs(args.save_dir, exist_ok=True)
    shutil.copy(__file__, args.save_dir)
    shutil.copy("models.py", args.save_dir)
    shutil.copy("dataloader.py", args.save_dir)
    tensorboard = SummaryWriter(args.save_dir)

    my_dataset = MyMapDatasetV5()
    my_train_dataset, my_valid_dataset = split_datasets(my_dataset)
    my_train_dataLoader = DataLoader(
        my_train_dataset, args.batch_size,
        shuffle=True, pin_memory=True, drop_last=True,
        num_workers=args.num_workers,
    )
    my_valid_dataLoader = DataLoader(
        my_valid_dataset, args.batch_size,
        shuffle=True, pin_memory=True, drop_last=True,
        num_workers=args.num_workers,
    )

    model        = MyModelV4().to(args.device)
    criterion    = torch.nn.MSELoss()
    optimizer    = torch.optim.Adam(model.parameters(), lr=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99995)

    best_valid_loss, best_valid_acc = np.inf, 0
    for epoch in range(1, args.epochs+1):
        print(f"\n{epoch}/{args.epochs}")
        model.train()
        train_loss, train_acc = train_epoch(model, my_train_dataLoader, criterion, optimizer, lr_scheduler, args.device)
        model.eval()
        valid_loss, valid_acc = valid_epoch(model, my_valid_dataLoader, criterion, args.device)
        tensorboard.add_scalar("0_Losses+LR/0_Train",     train_loss, epoch)
        tensorboard.add_scalar("0_Losses+LR/1_Valid",     valid_loss, epoch)
        tensorboard.add_scalar("0_Losses+LR/2_LR", get_lr(optimizer), epoch)
        tensorboard.add_scalar("1_Accuracies/0_Train", train_acc,  epoch)
        tensorboard.add_scalar("1_Accuracies/1_Valid", valid_acc,  epoch)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model, f"{args.save_dir}/best_valid_loss.pt")
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model, f"{args.save_dir}/best_valid_acc.pt")

    tensorboard.close()
    return



""" Execution """
if __name__ == "__main__":

    DEFAULT_DEVICE   = "cuda:0"
    DEFAULT_SAVE_DIR = f"logs/new/{time.strftime('%Y.%m.%d-%H.%M.%S', time.localtime())}"

    parser = argparse.ArgumentParser()
    parser.add_argument("-e",  "--epochs",      type=int, default=100)
    parser.add_argument("-bs", "--batch-size",  type=int, default=64)
    parser.add_argument("-nw", "--num-workers", type=int, default=4)
    parser.add_argument("-d",  "--device",      type=str, default=DEFAULT_DEVICE)
    parser.add_argument("-sd", "--save-dir",    type=str, default=DEFAULT_SAVE_DIR)

    args = parser.parse_args()
    main(args)