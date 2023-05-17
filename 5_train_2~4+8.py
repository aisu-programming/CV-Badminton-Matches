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
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from libs.dataloader import ClassificationDataset, BallTypeDataset
from libs.models import SingleOutputModel, BallTypeModel
from libs.misc import train_formal_list, train_informal_list



""" Constants """
TARGET_ID_2_STR = {
    "RoundHead" : "2_round_head",
    "Backhand"  : "3_backhand",
    "BallHeight": "4_ball_height",
    "BallType"  : "8_ball_type",
}



""" Classes """
class Metric():
    def __init__(self, length) -> None:
        self.length = length
        self.values = []

    def append(self, value) -> None:
        self.values.append(value)
        if len(self.values) > self.length: self.values.pop(0)
        return

    def avg(self) -> float:
        return np.average(self.values)



""" Functions """
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
    cm_df = pd.DataFrame(confusion_matrix, index=list(range(2)), columns=list(range(2)))
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
    batch_imgs, batch_kpts, batch_balls, batch_times, batch_bg_id, batch_hitter = batch_inputs
    batch_imgs   = batch_imgs.to(device)
    batch_kpts   = batch_kpts.to(device)
    batch_balls  = batch_balls.to(device)
    batch_times  = batch_times.to(device)
    batch_bg_id  = batch_bg_id.to(device)
    batch_hitter = batch_hitter.to(device)
    # print(batch_imgs.shape, batch_kpts.shape)
    batch_pred = model(batch_imgs, batch_kpts, batch_balls, batch_times, batch_bg_id, batch_hitter)
    return batch_pred


def train_epoch(model:SingleOutputModel, dataloader, criterion,
                optimizer, lr_scheduler, device, weight) -> None:
    
    loss_metric, acc_metric = Metric(50), Metric(50)
    acc_weight              = (np.array(weight)/sum(weight)).tolist()
    confusion_matrixs       = np.zeros((2, 2))

    pbar = tqdm(enumerate(dataloader), desc="[TRAIN]", total=len(dataloader))
    for batch_i, (batch_input, batch_truth) in pbar:
        model.zero_grad()
        batch_pred = forward_step(model, batch_input, device)
        batch_truth = batch_truth.to(device)

        # print(batch_pred.shape, batch_truth.shape)
        loss = criterion(batch_pred, batch_truth)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        loss_metric.append(loss.item())
        batch_truth = batch_truth.cpu().detach().numpy()
        batch_pred  = batch_pred.cpu().detach().numpy()

        acc = weighted_acc(np.argmax(batch_pred, axis=-1), batch_truth, acc_weight)
        acc_metric.append(acc)
        confusion_matrixs += confusion_matrix(batch_truth, np.argmax(batch_pred, axis=-1), labels=[0,1])  # , sample_weight=weight)
        pbar.set_description(f"[TRAIN] loss: {loss_metric.avg():.5f}, " + \
                             f"Acc: {acc_metric.avg()*100:.3f}%, " + \
                             f"LR: {get_lr(optimizer):.10f}")

    with np.printoptions(linewidth=150, formatter={'float': '{:5.03f}'.format, 'int': '{:2} '.format}):
        print("pred :", batch_pred[0], batch_pred[1], batch_pred[2])
        print("pred :", np.uint8(np.argmax(batch_pred, axis=-1))[:35])
        print("truth:", batch_truth[:35])
        print("corct:", np.array(["X", " "])[np.uint8(np.argmax(batch_pred, axis=-1)==batch_truth)[:35]])

    return loss_metric.avg(), acc_metric.avg(), confusion_matrixs


def valid_epoch(model:SingleOutputModel, dataloader,
                criterion, device, weight) -> None:
    
    loss_metric, acc_metric = Metric(1000), Metric(1000)
    acc_weight              = (np.array(weight)/sum(weight)).tolist()
    confusion_matrixs       = np.zeros((2, 2))

    pbar = tqdm(enumerate(dataloader), desc="[VALID]", total=len(dataloader))
    for batch_i, (batch_input, batch_truth) in pbar:

        model.zero_grad()
        batch_pred  = forward_step(model, batch_input, device)
        batch_truth = batch_truth.to(device)
        loss = criterion(batch_pred, batch_truth)
        loss_metric.append(loss.item())
        batch_truth = batch_truth.cpu().detach().numpy()
        batch_pred  = batch_pred.cpu().detach().numpy()

        acc = weighted_acc(np.argmax(batch_pred, axis=-1), batch_truth, acc_weight)
        acc_metric.append(acc)
        confusion_matrixs += confusion_matrix(batch_truth, np.argmax(batch_pred, axis=-1), labels=[0,1])  # , sample_weight=weight)
        pbar.set_description(f"[VALID] loss: {loss_metric.avg():.5f}, " + \
                             f"Acc: {acc_metric.avg()*100:.3f}%")

    with np.printoptions(linewidth=150, formatter={'float': '{:5.03f}'.format, 'int': '{:2} '.format}):
        print("pred :", batch_pred[0], batch_pred[1], batch_pred[2])
        print("pred :", np.uint8(np.argmax(batch_pred, axis=-1))[:35])
        print("truth:", batch_truth[:35])
        print("corct:", np.array(["X", " "])[np.uint8(np.argmax(batch_pred, axis=-1)==batch_truth)[:35]])
        
    return loss_metric.avg(), acc_metric.avg(), confusion_matrixs


def main(args) -> None:

    os.makedirs(args.save_dir, exist_ok=True)
    shutil.copy(__file__, args.save_dir)
    shutil.copy("models.py", args.save_dir)
    shutil.copy("dataloader.py", args.save_dir)
    tensorboard = SummaryWriter(args.save_dir)

    if   args.mode ==   "formal": video_id_list = train_formal_list.copy()
    elif args.mode == "informal": video_id_list = train_informal_list.copy()
    else                        : video_id_list = list(range(1, 800+1))
    random.shuffle(video_id_list)
    train_video_id_list = video_id_list[:int(len(video_id_list)*0.8)]
    valid_video_id_list = video_id_list[int(len(video_id_list)*0.8):]
    print(valid_video_id_list)
    with open(f"{args.save_dir}/valid_video_id_list.py", 'w') as file:
        json.dump(valid_video_id_list, file, indent=4)

    if args.target != "BallType":
        my_train_dataset = ClassificationDataset(args.length, train_video_id_list, args.target)
        my_valid_dataset = ClassificationDataset(args.length, valid_video_id_list, args.target)
    else:
        my_train_dataset = BallTypeDataset(args.length, train_video_id_list)
        my_valid_dataset = BallTypeDataset(args.length, valid_video_id_list)
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

    print("ground_truth_count:", my_train_dataset.ground_truth_count)
    train_btc_avg      = sum(my_train_dataset.ground_truth_count) / len(my_train_dataset.ground_truth_count)
    train_weight       = [ (train_btc_avg/btc) for btc in my_train_dataset.ground_truth_count ]
    train_weight_torch = torch.from_numpy(np.array(train_weight)).float().to(args.device)
    valid_btc_avg      = sum(my_valid_dataset.ground_truth_count) / len(my_valid_dataset.ground_truth_count)
    valid_weight       = [ (valid_btc_avg/btc) for btc in my_valid_dataset.ground_truth_count ]

    # output_dim   = 2 if args.target != "BallType" else 9
    if args.target != "BallType":
        model    = SingleOutputModel(length=args.length, output_dim=2, softmax=True).to(args.device)
    else:
        model    = BallTypeModel(length=args.length, softmax=True).to(args.device)
    criterion    = torch.nn.CrossEntropyLoss(weight=train_weight_torch)
    optimizer    = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)

    best_valid_loss, best_valid_acc = np.inf, 0
    for epoch in range(1, args.epochs+1):
        print(f"\n{epoch}/{args.epochs}")
        model = model.train()
        train_loss, train_acc, train_cm = \
            train_epoch(model, my_train_dataLoader, criterion, optimizer, lr_scheduler, args.device, train_weight)
        model = model.eval()
        valid_loss, valid_acc, valid_cm = \
            valid_epoch(model, my_valid_dataLoader, criterion, args.device, valid_weight)
        tensorboard.add_scalar("0_Losses+LR/0_Train",     train_loss, epoch)
        tensorboard.add_scalar("0_Losses+LR/1_Valid",     valid_loss, epoch)
        tensorboard.add_scalar("0_Losses+LR/2_LR", get_lr(optimizer), epoch)
        tensorboard.add_scalar("1_Accuracies/0_Train",     train_acc, epoch)
        tensorboard.add_scalar("1_Accuracies/1_Valid",     valid_acc, epoch)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            plot_confusion_matrix(train_cm, f"{args.save_dir}/best_valid_loss_train_cm.png", "Train Confusion Matirx at Best Valid Loss")
            plot_confusion_matrix(valid_cm, f"{args.save_dir}/best_valid_loss_valid_cm.png", "Valid Confusion Matirx at Best Valid Loss")
            torch.save(model, f"{args.save_dir}/best_valid_loss.pt")
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            plot_confusion_matrix(train_cm, f"{args.save_dir}/best_valid_acc_train_cm.png", "Train Confusion Matirx at Best Valid Acc")
            plot_confusion_matrix(valid_cm, f"{args.save_dir}/best_valid_acc_valid_cm.png", "Valid Confusion Matirx at Best Valid Acc")
            torch.save(model, f"{args.save_dir}/best_valid_acc.pt")

    tensorboard.close()
    return



""" Execution """
if __name__ == "__main__":
    
    # Targt options: [ "RoundHead", "Backhand", "BallHeight", "BallType" ]
    DEFAULT_TARGET   = "BallType"

    DEFAULT_MODE     = "all"
    DEFAULT_DEVICE   = "cuda:1"
    DEFAULT_LENGTH     = 11 if DEFAULT_TARGET != "BallType" else 45 # BallType (Recommended 45 / Maximum 51)

    parser = argparse.ArgumentParser()
    parser.add_argument("-t",  "--target",        type=str,   default=DEFAULT_TARGET)
    parser.add_argument("-m",  "--mode",          type=str,   default=DEFAULT_MODE)
    parser.add_argument("-e",  "--epochs",        type=int,   default=100)
    parser.add_argument("-lr", "--learning-rate", type=float, default=3e-3)
    parser.add_argument("-ld", "--lr-decay",      type=float, default=0.9985)
    parser.add_argument("-bs", "--batch-size",    type=int,   default=128)
    parser.add_argument("-nw", "--num-workers",   type=int,   default=4)
    parser.add_argument("-l",  "--length",        type=int,   default=DEFAULT_LENGTH)  
    parser.add_argument("-d",  "--device",        type=str,   default=DEFAULT_DEVICE)

    args = parser.parse_args()
    assert args.target in [ "RoundHead", "Backhand", "BallHeight", "BallType" ]
    assert args.mode   in [ "all", "formal", "informal" ]
    assert (args.length-1) % 2 == 0
    args.save_dir = f"logs/{args.mode}/{TARGET_ID_2_STR[args.target]}/" + \
                        time.strftime('%Y.%m.%d-%H.%M.%S', time.localtime())
    main(args)