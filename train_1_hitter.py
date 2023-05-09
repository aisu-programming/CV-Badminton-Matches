""" Libraries """
import os
import time
import json
import torch
import shutil
import random
import argparse
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Any, Iterator, Tuple
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataloader import HitterDataset, split_datasets
from models import HitterModel
from misc import train_formal_list



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

    def avg(self) -> float:
        return np.average(self.values)


def weighted_acc(pred, truth, weight):
    acc = 0.0
    # assert np.max(np.max(pred), np.max(truth)) <= len(weight)
    for wid, wt in enumerate(weight):
        acc += (np.logical_and(pred==wid, pred==truth).sum()/np.max((1, (truth==wid).sum()))) * wt
    return acc


def get_lr(optimizer) -> float:
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    

def plot_confusion_matrix(confusion_matrix, filename, title):
    cm_df = pd.DataFrame(confusion_matrix, index=list(range(3)), columns=list(range(3)))
    plt.figure(figsize=(6, 5))
    s = sn.heatmap(cm_df, annot=True)
    s.set_xlabel("prediction", fontsize=10)
    s.set_ylabel("truth", fontsize=10)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return


def forward_step(model, batch_inputs, device) -> torch.Tensor:
    batch_imgs, batch_kpts, batch_balls, batch_times, batch_bg_id = batch_inputs
    batch_imgs  = batch_imgs.to(device)
    batch_kpts  = batch_kpts.to(device)
    batch_balls = batch_balls.to(device)
    batch_times = batch_times.to(device)
    batch_bg_id = batch_bg_id.to(device)
    # print(batch_imgs.shape, batch_kpts.shape)
    batch_pred = model(batch_imgs, batch_kpts, batch_balls, batch_times, batch_bg_id)
    return batch_pred


def train_epoch(model, dataloader, criterion, optimizer, lr_scheduler, device, length, weight) -> None:
    loss_metric = Metric(50)
    oacc_metric = Metric(50)  # Overall ACC
    nacc_metric = Metric(50)  # Narrow  ACC
    acc_weight  = (np.array(weight)/sum(weight)).tolist()
    ocms = np.zeros((3, 3))   # Overall confusion matrix
    ncms = np.zeros((3, 3))   # Narrow  confusion matrix

    pbar = tqdm(enumerate(dataloader), desc="[TRAIN]", total=len(dataloader))
    for batch_i, (batch_input, batch_truth) in pbar:
        model.zero_grad()
        batch_pred = forward_step(model, batch_input, device)
        batch_truth = batch_truth.to(device)

        # print(batch_pred.shape, batch_truth.shape)
        loss = criterion(torch.permute(batch_pred, dims=(0, 2, 1)), batch_truth)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        loss_metric.append(loss.item())
        batch_truth = batch_truth.cpu().detach().numpy()
        batch_pred  = batch_pred.cpu().detach().numpy()

        ao_batch_pred = np.argmax(batch_pred, axis=-1)                                    # argmaxed overall batch_pred
        an_batch_pred = np.argmax(batch_pred, axis=-1)[:, (length//2-2):(length//2+2)+1]  # argmaxed  narrow batch_pred
        n_batch_truth =                    batch_truth[:, (length//2-2):(length//2+2)+1]  #           narrow batch_truth
        oacc = weighted_acc(ao_batch_pred, batch_truth, acc_weight)
        oacc_metric.append(oacc)
        nacc = weighted_acc(an_batch_pred, n_batch_truth, acc_weight)
        nacc_metric.append(nacc)

        for bt, bp in zip(batch_truth, ao_batch_pred):
            ocms += confusion_matrix(bt, bp, labels=[0,1,2])  # , sample_weight=weight)
        for bt, bp in zip(n_batch_truth, an_batch_pred):
            ncms += confusion_matrix(bt, bp, labels=[0,1,2])  # , sample_weight=weight)

        pbar.set_description(f"[TRAIN] loss: {loss_metric.avg():.5f}, " + \
                             f"Overall Acc: {oacc_metric.avg()*100:.3f}%, " + \
                             f"Narrow Acc: {nacc_metric.avg()*100:.3f}%, " + \
                             f"LR: {get_lr(optimizer):.10f}")

    with np.printoptions(linewidth=150, formatter={'float': '{:5.03f}'.format, 'int': '{:2} '.format}):
        print("pred :", batch_pred[0, 0])
        print("pred :", np.round(np.argmax(batch_pred, axis=-1)[0]))
        print("truth:", batch_truth[0])
        print("corct:", np.array(["X", " "])[np.uint8(np.argmax(batch_pred, axis=-1)==batch_truth)[0]])

    return loss_metric.avg(), oacc_metric.avg(), nacc_metric.avg(), ocms, ncms


def valid_epoch(model, dataloader, criterion, device, length, weight) -> None:
    loss_metric = Metric(1000)
    oacc_metric = Metric(1000)  # Overall ACC
    nacc_metric = Metric(1000)  # Narrow  ACC
    acc_weight  = (np.array(weight)/sum(weight)).tolist()
    ocms = np.zeros((3, 3))     # Overall confusion matrix
    ncms = np.zeros((3, 3))     # Narrow  confusion matrix

    pbar = tqdm(enumerate(dataloader), desc="[VALID]", total=len(dataloader))
    for batch_i, (batch_input, batch_truth) in pbar:

        batch_pred  = forward_step(model, batch_input, device)
        batch_truth = batch_truth.to(device)
        loss = criterion(torch.permute(batch_pred, dims=(0, 2, 1)), batch_truth)
        loss_metric.append(loss.item())
        batch_truth = batch_truth.cpu().detach().numpy()
        batch_pred  = batch_pred.cpu().detach().numpy()

        ao_batch_pred = np.argmax(batch_pred, axis=-1)                                    # argmaxed overall batch_pred
        an_batch_pred = np.argmax(batch_pred, axis=-1)[:, (length//2-2):(length//2+2)+1]  # argmaxed  narrow batch_pred
        n_batch_truth =                    batch_truth[:, (length//2-2):(length//2+2)+1]  #           narrow batch_truth
        oacc = weighted_acc(ao_batch_pred, batch_truth, acc_weight)
        oacc_metric.append(oacc)
        nacc = weighted_acc(an_batch_pred, n_batch_truth, acc_weight)
        nacc_metric.append(nacc)

        for bt, bp in zip(batch_truth, ao_batch_pred):
            ocms += confusion_matrix(bt, bp, labels=[0,1,2])  # , sample_weight=weight)
        for bt, bp in zip(n_batch_truth, an_batch_pred):
            ncms += confusion_matrix(bt, bp, labels=[0,1,2])  # , sample_weight=weight)

        pbar.set_description(f"[VALID] loss: {loss_metric.avg():.5f}, " + \
                             f"Overall Acc: {oacc_metric.avg()*100:.3f}%, " + \
                             f"Narrow Acc: {nacc_metric.avg()*100:.3f}%")

    with np.printoptions(linewidth=150, formatter={'float': '{:5.03f}'.format, 'int': '{:2} '.format}):
        print("pred :", batch_pred[0, 0])
        print("pred :", np.round(np.argmax(batch_pred, axis=-1)[0]))
        print("truth:", batch_truth[0])
        print("corct:", np.array(["X", " "])[np.uint8(np.argmax(batch_pred, axis=-1)==batch_truth)[0]])
        
    return loss_metric.avg(), oacc_metric.avg(), nacc_metric.avg(), ocms, ncms


def main(args) -> None:

    os.makedirs(args.save_dir, exist_ok=True)
    shutil.copy(__file__, args.save_dir)
    shutil.copy("models.py", args.save_dir)
    shutil.copy("dataloader.py", args.save_dir)
    tensorboard = SummaryWriter(args.save_dir)

    video_id_list = train_formal_list.copy()
    random.shuffle(video_id_list)
    train_video_id_list = video_id_list[:int(len(video_id_list)*0.8)]
    valid_video_id_list = video_id_list[int(len(video_id_list)*0.8):]
    print(valid_video_id_list)
    with open(f"{args.save_dir}/valid_video_id_list.py", 'w') as file:
        json.dump(valid_video_id_list, file, indent=4)

    my_train_dataset = HitterDataset(args.length, train_video_id_list)
    my_valid_dataset = HitterDataset(args.length, valid_video_id_list)
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

    print("ball_type_count:", my_train_dataset.ball_type_count)
    train_btc_avg      = sum(my_train_dataset.ball_type_count) / len(my_train_dataset.ball_type_count)
    train_weight       = [ (train_btc_avg/btc) for btc in my_train_dataset.ball_type_count ]
    train_weight_torch = torch.from_numpy(np.array(train_weight)).float().to(args.device)

    valid_btc_avg      = sum(my_valid_dataset.ball_type_count) / len(my_valid_dataset.ball_type_count)
    valid_weight       = [ (valid_btc_avg/btc) for btc in my_valid_dataset.ball_type_count ]

    model        = HitterModel(args.length).to(args.device)
    criterion    = torch.nn.CrossEntropyLoss(weight=train_weight_torch)
    optimizer    = torch.optim.Adam(model.parameters(), lr=7e-4)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9993)

    best_valid_loss, best_valid_oacc, best_valid_nacc = np.inf, 0, 0
    for epoch in range(1, args.epochs+1):
        print(f"\n{epoch}/{args.epochs}")
        model = model.train()
        train_loss, train_oacc, train_nacc, train_ocm, train_ncm = \
            train_epoch(model, my_train_dataLoader, criterion, optimizer, lr_scheduler, args.device, args.length, train_weight)
        model = model.eval()
        valid_loss, valid_oacc, valid_nacc, valid_ocm, valid_ncm = \
            valid_epoch(model, my_valid_dataLoader, criterion, args.device, args.length, valid_weight)
        tensorboard.add_scalar("0_Losses+LR/0_Train",          train_loss, epoch)
        tensorboard.add_scalar("0_Losses+LR/1_Valid",          valid_loss, epoch)
        tensorboard.add_scalar("0_Losses+LR/2_LR",      get_lr(optimizer), epoch)
        tensorboard.add_scalar("1_Overall_Accuracies/0_Train", train_oacc, epoch)
        tensorboard.add_scalar("1_Overall_Accuracies/1_Valid", valid_oacc, epoch)
        tensorboard.add_scalar("2_Narrow_Accuracies/0_Train",  train_nacc, epoch)
        tensorboard.add_scalar("2_Narrow_Accuracies/1_Valid",  valid_nacc, epoch)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            plot_confusion_matrix(train_ocm, f"{args.save_dir}/best_valid_loss_train_ocm.png", "Train Overall Confusion Matirx at Best Valid Loss")
            plot_confusion_matrix(valid_ocm, f"{args.save_dir}/best_valid_loss_valid_ocm.png", "Valid Overall Confusion Matirx at Best Valid Loss")
            plot_confusion_matrix(train_ncm, f"{args.save_dir}/best_valid_loss_train_ncm.png", "Train Narrow Confusion Matirx at Best Valid Loss")
            plot_confusion_matrix(valid_ncm, f"{args.save_dir}/best_valid_loss_valid_ncm.png", "Valid Narrow Confusion Matirx at Best Valid Loss")
            torch.save(model, f"{args.save_dir}/best_valid_loss.pt")
        if valid_oacc > best_valid_oacc:
            best_valid_oacc = valid_oacc
            plot_confusion_matrix(train_ocm, f"{args.save_dir}/best_valid_oacc_train_ocm.png", "Train Overall Confusion Matirx at Best Valid Overall Acc")
            plot_confusion_matrix(valid_ocm, f"{args.save_dir}/best_valid_oacc_valid_ocm.png", "Valid Overall Confusion Matirx at Best Valid Overall Acc")
            plot_confusion_matrix(train_ncm, f"{args.save_dir}/best_valid_oacc_train_ncm.png", "Train Narrow Confusion Matirx at Best Valid Overall Acc")
            plot_confusion_matrix(valid_ncm, f"{args.save_dir}/best_valid_oacc_valid_ncm.png", "Valid Narrow Confusion Matirx at Best Valid Overall Acc")
            torch.save(model, f"{args.save_dir}/best_valid_oacc.pt")
        if valid_nacc > best_valid_nacc:
            best_valid_nacc = valid_nacc
            plot_confusion_matrix(train_ocm, f"{args.save_dir}/best_valid_nacc_train_ocm.png", "Train Overall Confusion Matirx at Best Valid Narrow Acc")
            plot_confusion_matrix(valid_ocm, f"{args.save_dir}/best_valid_nacc_valid_ocm.png", "Valid Overall Confusion Matirx at Best Valid Narrow Acc")
            plot_confusion_matrix(train_ncm, f"{args.save_dir}/best_valid_nacc_train_ncm.png", "Train Narrow Confusion Matirx at Best Valid Narrow Acc")
            plot_confusion_matrix(valid_ncm, f"{args.save_dir}/best_valid_nacc_valid_ncm.png", "Valid Narrow Confusion Matirx at Best Valid Narrow Acc")
            torch.save(model, f"{args.save_dir}/best_valid_nacc.pt")

    tensorboard.close()
    return



""" Execution """
if __name__ == "__main__":

    DEFAULT_MODE     = "train"
    DEFAULT_DEVICE   = "cuda:0"
    DEFAULT_SAVE_DIR = f"logs/1_hitter/{time.strftime('%Y.%m.%d-%H.%M.%S', time.localtime())}"

    parser = argparse.ArgumentParser()
    parser.add_argument("-m",  "--mode",        type=str, default=DEFAULT_MODE)
    parser.add_argument("-e",  "--epochs",      type=int, default=200)
    parser.add_argument("-l",  "--length",      type=int, default=31)
    parser.add_argument("-bs", "--batch-size",  type=int, default=80)
    parser.add_argument("-nw", "--num-workers", type=int, default=4)
    parser.add_argument("-d",  "--device",      type=str, default=DEFAULT_DEVICE)
    parser.add_argument("-sd", "--save-dir",    type=str, default=DEFAULT_SAVE_DIR)

    args = parser.parse_args()
    assert args.mode in [ "train", "valid" ]
    assert (args.length-1) % 2 == 0
    main(args)