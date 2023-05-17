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

from libs.dataloader import LocationDataset
from libs.models import SingleOutputModel
from libs.misc import train_formal_list, train_informal_list



""" Constants """
TARGET_ID_2_STR = {
        "Landing"         : "5_landing",
        "HitterLocation"  : "6_hitter_location",
        "DefenderLocation": "7_defender_location",
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


class MyLoss(torch.nn.Module):
    def __init__(self) -> None:
        super(MyLoss, self).__init__()
    
    def forward(self, pred, truth) -> torch.Tensor:
        losses = torch.sum((pred-truth)**2, dim=-1)**0.5
        assert losses.shape == truth.shape[:-1]
        return torch.mean(losses)



""" Functions """
def calculate_acc(pred, truth, target):
    acc = np.sum((pred-truth)**2, axis=-1)**0.5
    assert acc.shape == truth.shape[:-1]
    threshold = 6 if target == "Landing" else 10
    acc = np.average(acc<threshold)
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
    # batch_imgs, batch_kpt_imgs, batch_kpts, batch_balls, batch_times, batch_bg_id, batch_hitter = batch_inputs
    batch_kpt_imgs, batch_kpts, batch_balls, batch_times, batch_bg_id, batch_hitter = batch_inputs
    # batch_imgs     = batch_imgs.to(device)
    batch_kpt_imgs = batch_kpt_imgs.to(device)
    batch_kpts     = batch_kpts.to(device)
    batch_balls    = batch_balls.to(device)
    batch_times    = batch_times.to(device)
    batch_bg_id    = batch_bg_id.to(device)
    batch_hitter   = batch_hitter.to(device)
    # print(batch_imgs.shape, batch_kpts.shape)
    # batch_pred = model(batch_imgs, batch_kpt_imgs, batch_kpts, batch_balls, batch_times, batch_bg_id, batch_hitter)
    batch_pred = model(batch_kpt_imgs, batch_kpts, batch_balls, batch_times, batch_bg_id, batch_hitter)
    return batch_pred


def train_epoch(model:SingleOutputModel, dataloader, criterion,
                optimizer, lr_scheduler, device, target) -> None:
    
    loss_metric, acc_metric = Metric(20), Metric(20)
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

        batch_pred[:, 0]  *= 1280
        batch_pred[:, 1]  *=  720
        batch_truth[:, 0] *= 1280
        batch_truth[:, 1] *=  720
        acc = calculate_acc(batch_pred, batch_truth, target)
        acc_metric.append(acc)
        pbar.set_description(f"[TRAIN] loss: {loss_metric.avg():.5f}, " + \
                             f"Acc: {acc_metric.avg()*100:.3f}%, " + \
                             f"LR: {get_lr(optimizer):.10f}")

    with np.printoptions(linewidth=150, formatter={"float": "{:7.02f}".format}):
        threshold = 6 if target == "Landing" else 10
        print("pred :", batch_pred[0], batch_pred[1], batch_pred[2], batch_pred[3], batch_pred[4])
        print("truth:", batch_truth[0], batch_truth[1], batch_truth[2], batch_truth[3], batch_truth[4])
        print("corct:", np.array(['_', 'O'])[np.uint8(np.sum((batch_pred-batch_truth)**2, axis=-1)**0.5<threshold)[:30]])

    return loss_metric.avg(), acc_metric.avg()


def valid_epoch(model:SingleOutputModel, dataloader,
                criterion, device, target) -> None:
    
    loss_metric, acc_metric = Metric(1000), Metric(1000)
    pbar = tqdm(enumerate(dataloader), desc="[VALID]", total=len(dataloader))
    for batch_i, (batch_input, batch_truth) in pbar:

        model.zero_grad()
        batch_pred  = forward_step(model, batch_input, device)
        batch_truth = batch_truth.to(device)
        loss = criterion(batch_pred, batch_truth)
        loss_metric.append(loss.item())
        batch_truth = batch_truth.cpu().detach().numpy()
        batch_pred  = batch_pred.cpu().detach().numpy()

        batch_pred[:, 0]  *= 1280
        batch_pred[:, 1]  *=  720
        batch_truth[:, 0] *= 1280
        batch_truth[:, 1] *=  720
        acc = calculate_acc(batch_pred, batch_truth, target)
        acc_metric.append(acc)
        pbar.set_description(f"[VALID] loss: {loss_metric.avg():.5f}, " + \
                             f"Acc: {acc_metric.avg()*100:.3f}%")

    with np.printoptions(linewidth=150, formatter={"float": "{:7.02f}".format}):
        threshold = 6 if target == "Landing" else 10
        print("pred :", batch_pred[0], batch_pred[1], batch_pred[2], batch_pred[3], batch_pred[4])
        print("truth:", batch_truth[0], batch_truth[1], batch_truth[2], batch_truth[3], batch_truth[4])
        print("corct:", np.array(['_', 'O'])[np.uint8(np.sum((batch_pred-batch_truth)**2, axis=-1)**0.5<threshold)[:30]])
        
    return loss_metric.avg(), acc_metric.avg()


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
    train_video_id_list = sorted(video_id_list[:int(len(video_id_list)*0.8)])
    valid_video_id_list = sorted(video_id_list[int(len(video_id_list)*0.8):])
    print(valid_video_id_list)
    with open(f"{args.save_dir}/valid_video_id_list.py", 'w') as file:
        json.dump(valid_video_id_list, file, indent=4)

    my_train_dataset = LocationDataset(args.length, train_video_id_list, args.target)
    my_valid_dataset = LocationDataset(args.length, valid_video_id_list, args.target)
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

    model        = SingleOutputModel(length=args.length, output_dim=2, softmax=False, sigmoid=False).to(args.device)
    criterion    = MyLoss()
    optimizer    = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)

    best_valid_loss, best_valid_acc = np.inf, 0
    for epoch in range(1, args.epochs+1):
        print(f"\n{epoch}/{args.epochs}")
        model = model.train()
        train_loss, train_acc = \
            train_epoch(model, my_train_dataLoader, criterion, optimizer, lr_scheduler, args.device, args.target)
        model = model.eval()
        valid_loss, valid_acc = \
            valid_epoch(model, my_valid_dataLoader, criterion, args.device, args.target)
        tensorboard.add_scalar("0_Losses+LR/0_Train",     train_loss, epoch)
        tensorboard.add_scalar("0_Losses+LR/1_Valid",     valid_loss, epoch)
        tensorboard.add_scalar("0_Losses+LR/2_LR", get_lr(optimizer), epoch)
        tensorboard.add_scalar("1_Accuracies/0_Train",     train_acc, epoch)
        tensorboard.add_scalar("1_Accuracies/1_Valid",     valid_acc, epoch)

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

    DEFAULT_TARGET = "Landing"
    DEFAULT_MODE   = "all"
    DEFAULT_DEVICE = "cuda:0"

    parser = argparse.ArgumentParser()
    parser.add_argument("-t",  "--target",        type=str,   default=DEFAULT_TARGET)
    parser.add_argument("-m",  "--mode",          type=str,   default=DEFAULT_MODE)
    parser.add_argument("-e",  "--epochs",        type=int,   default=150)
    parser.add_argument("-lr", "--learning-rate", type=float, default=5e-4)
    parser.add_argument("-ld", "--lr-decay",      type=float, default=0.9975)
    parser.add_argument("-bs", "--batch-size",    type=int,   default=80)
    parser.add_argument("-nw", "--num-workers",   type=int,   default=4)
    parser.add_argument("-l",  "--length",        type=int,   default=7)
    parser.add_argument("-d",  "--device",        type=str,   default=DEFAULT_DEVICE)

    args = parser.parse_args()
    assert args.target in [ "Landing", "HitterLocation", "DefenderLocation" ]
    assert args.mode   in [ "all", "formal", "informal" ]
    assert (args.length-1) % 2 == 0
    args.save_dir = f"logs/{args.mode}/{TARGET_ID_2_STR[args.target]}/" + \
                        time.strftime('%Y.%m.%d-%H.%M.%S', time.localtime())
    main(args)