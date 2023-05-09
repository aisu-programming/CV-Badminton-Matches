import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

from misc import train_formal_list


def equation_1st(x, a, b):
    return a* x + b

def equation_2nd(x, a, b, c):
    return a* x**2 + b* x + c

def equation_3rd(x, a, b, c, d):
    return a* x**3 + b* x**2 + c* x + d

def equation_Nth(x, *a):
    ret = 0
    for nth in range(len(a)):
        ret = a[nth]* x**(len(a)-nth-1)
    return ret

def fourier(x, *a):
    tau = 1500
    ret = a[0] * np.cos(np.pi / tau * x)
    for deg in range(1, len(a)):
        ret += a[deg] * np.cos((deg+1) * np.pi / tau * x)
    return ret

def plot_each_videos_trajectories():
    os.makedirs("analysis/each_videos_trajectories", exist_ok=True)
    for video_id in tqdm(train_formal_list):

        hit_frames = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_S2.csv")["HitFrame"].values
        pose_df = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_pose.csv")
        frame_count = len(pose_df.values)
        ball_df = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_ball_33_adj.csv")

        adj_ball_diff_df    = ball_df.replace(np.nan, 0)
        adj_ball_diff_frame = adj_ball_diff_df[["Frame"]].values.squeeze()
        adj_ball_x_tmp      = adj_ball_diff_df[["Adjusted X"]].values.squeeze()
        adj_ball_x_shift    = adj_ball_x_tmp[1:]
        adj_ball_x_tmp      = adj_ball_x_tmp[:-1]
        adj_nonzeros        = (adj_ball_x_tmp * adj_ball_x_shift) != 0
        adj_ball_diff_frame = adj_ball_diff_frame[:-1][adj_nonzeros]
        adj_ball_x_shift    = adj_ball_x_shift[adj_nonzeros]
        adj_ball_x_tmp      = adj_ball_x_tmp[adj_nonzeros]
        adj_ball_x_diff     = adj_ball_x_shift - adj_ball_x_tmp

        ori_ball_df    = ball_df.replace(np.nan, 0)
        ori_ball_df    = ori_ball_df[(ori_ball_df["X"]!=0) | (ori_ball_df["Y"]!=0)]
        ori_ball_frame = ori_ball_df[["Frame"]].values.squeeze()
        # ori_ball_x     = ori_ball_df[["X"]].values.squeeze()
        ori_ball_y     = ori_ball_df[["Y"]].values.squeeze()

        adj_ball_df    = ball_df.replace(np.nan, 0)
        adj_ball_df    = adj_ball_df[(adj_ball_df["Adjusted X"]!=0) | (adj_ball_df["Adjusted Y"]!=0)]
        adj_ball_frame = adj_ball_df[["Frame"]].values.squeeze()
        adj_ball_x     = adj_ball_df[["Adjusted X"]].values.squeeze()
        adj_ball_y     = adj_ball_df[["Adjusted Y"]].values.squeeze()

        adj_ball_y_peaks,   properties = find_peaks(adj_ball_y,      distance=7, prominence=12)
        adj_ball_y_r_peaks, properties = find_peaks(adj_ball_y*(-1), distance=7, prominence=12)  # reverse
        adj_ball_x_peaks,   properties = find_peaks(adj_ball_x,      distance=7, prominence=12, width=(0, 30))
        adj_ball_x_r_peaks, properties = find_peaks(adj_ball_x*(-1), distance=7, prominence=12, width=(0, 30))  # reverse

        adj_ball_x_diff_abs = abs(adj_ball_x_diff)
        adj_ball_x_diff_abs_peaks, properties = find_peaks(adj_ball_x_diff_abs, distance=7, prominence=12)

        fig = plt.figure(figsize=(20, 12))
        ax = fig.add_subplot(4, 2, 1)
        ax.scatter(adj_ball_x[adj_ball_y_peaks], adj_ball_y[adj_ball_y_peaks], label="Peak", c='r', s=20)
        ax.scatter(adj_ball_x, adj_ball_y, c='g', s=3)
        ax.legend()
        ax.set_title("Adj X / Y")
        ax.invert_yaxis()
        ax.grid(axis='y')

        ax = fig.add_subplot(4, 2, 2)
        ax.scatter(adj_ball_frame[adj_ball_x_peaks], adj_ball_x[adj_ball_x_peaks], label="Peak", c='r', s=20)
        ax.scatter(adj_ball_frame, adj_ball_x, c='g', s=3)
        ax.set_xlim(-10, frame_count+10)
        for hf in hit_frames: ax.axvline(x=hf, c='b', ls="--", lw=1)
        ax.legend()
        ax.set_title("Adj Frame / X (with peaks)")
        ax.grid(axis='y')

        ax = fig.add_subplot(4, 2, 3)
        ax.scatter(ori_ball_frame, ori_ball_y, s=3, c='g')
        ax.set_xlim(-10, frame_count+10)
        for hf in hit_frames: ax.axvline(x=hf, c='b', ls="--", lw=1)
        ax.set_title("Ori Frame / Y")
        ax.invert_yaxis()
        ax.grid(axis='y')

        ax = fig.add_subplot(4, 2, 4)
        ax.scatter(adj_ball_frame[adj_ball_x_r_peaks], adj_ball_x[adj_ball_x_r_peaks], label="Peak", c='r', s=20)
        ax.scatter(adj_ball_frame, adj_ball_x, c='g', s=3)
        ax.set_xlim(-10, frame_count+10)
        for hf in hit_frames: ax.axvline(x=hf, c='b', ls="--", lw=1)
        ax.legend()
        ax.set_title("Adj Frame / X (with reversed peaks)")
        ax.grid(axis='y')

        ax = fig.add_subplot(4, 2, 5)
        ax.scatter(adj_ball_frame[adj_ball_y_peaks], adj_ball_y[adj_ball_y_peaks], label="Peak", c='r', s=20)
        ax.scatter(adj_ball_frame, adj_ball_y, s=3, c='g')
        ax.set_xlim(-10, frame_count+10)
        for hf in hit_frames: ax.axvline(x=hf, c='b', ls="--", lw=1)
        ax.legend()
        ax.set_title("Adj Frame / Y (with peaks)")
        ax.invert_yaxis()
        ax.grid(axis='y')

        ax = fig.add_subplot(4, 2, 6)
        ax.scatter(adj_ball_diff_frame, adj_ball_x_diff, s=3, c='g')
        ax.set_xlim(-10, frame_count+10)
        for hf in hit_frames: ax.axvline(x=hf, c='b', ls="--", lw=1)
        ax.set_title("Adj Frame / X diff")
        ax.grid(axis='y')

        ax = fig.add_subplot(4, 2, 7)
        ax.scatter(adj_ball_frame[adj_ball_y_r_peaks], adj_ball_y[adj_ball_y_r_peaks], label="Peak", c='r', s=20)
        ax.scatter(adj_ball_frame, adj_ball_y, s=3, c='g')
        ax.set_xlim(-10, frame_count+10)
        for hf in hit_frames: ax.axvline(x=hf, c='b', ls="--", lw=1)
        ax.legend()
        ax.set_title("Adj Frame / Y (with reversed peaks)")
        ax.invert_yaxis()
        ax.grid(axis='y')

        ax = fig.add_subplot(4, 2, 8)
        ax.scatter(adj_ball_diff_frame[adj_ball_x_diff_abs_peaks], adj_ball_x_diff_abs[adj_ball_x_diff_abs_peaks], label="Peak", c='r', s=20)
        ax.scatter(adj_ball_diff_frame, adj_ball_x_diff_abs, s=3, c='g')
        ax.set_xlim(-10, frame_count+10)
        for hf in hit_frames: ax.axvline(x=hf, c='b', ls="--", lw=1)
        ax.legend()
        ax.set_title("Adj Frame / X diff abs (with peaks)")
        ax.grid(axis='y')

        if True:
        
            # adj_ball_y_shift = np.concatenate([adj_ball_y[1:], [adj_ball_y[-1]]])
            # adj_ball_y_diff  = adj_ball_y_shift - adj_ball_y
            # adj_ball_y_frame = adj_ball_frame[(adj_ball_y_shift != 0) & (adj_ball_y != 0)]
            # adj_ball_y_diff  = adj_ball_y_diff[(adj_ball_y_shift != 0) & (adj_ball_y != 0)]
            # adj_ball_x_shift = np.concatenate([adj_ball_x[1:], [adj_ball_x[-1]]])
            # adj_ball_x_diff  = adj_ball_x_shift - adj_ball_x
            # adj_ball_x_frame = adj_ball_frame[(adj_ball_x_shift != 0) & (adj_ball_x != 0)]
            # adj_ball_x_diff  = adj_ball_x_diff[(adj_ball_x_shift != 0) & (adj_ball_x != 0)]
            # adj_ball_diff    = (adj_ball_x_diff**2 + adj_ball_y_diff**2) **0.5
            # adj_ball_diff_frame = adj_ball_x_frame[adj_ball_diff>1]
            # adj_ball_diff       = adj_ball_diff[adj_ball_diff>1]

            # ax = fig.add_subplot(3, 3, 4)
            # for did in range(len(adj_ball_diff)-1):
            #     if adj_ball_diff_frame[did+1] - adj_ball_diff_frame[did] > 20: continue
            #     ax.plot(adj_ball_diff_frame[did:did+2], adj_ball_diff[did:did+2], marker='o', label="Adj", c='g', markersize=2, lw=0.5)
            # ax.set_xlim(-10, frame_count+10)
            # for hf in hit_frames:
            #     ax.axvline(x=hf, c='b', ls="--", lw=1)
            # ax.grid(axis='y')

            # ax = fig.add_subplot(3, 3, 5)
            # for yid in range(len(adj_ball_y_frame)-1):
            #     if adj_ball_y_frame[yid+1] - adj_ball_y_frame[yid] > 20: continue
            #     ax.plot(adj_ball_y_frame[yid:yid+2], adj_ball_y_diff[yid:yid+2], marker='o', label="Adj", c='g', markersize=2, lw=0.5)
            # ax.set_xlim(-10, frame_count+10)
            # for hf in hit_frames:
            #     ax.axvline(x=hf, c='b', ls="--", lw=1)
            # ax.grid(axis='y')

            # ax = fig.add_subplot(3, 3, 6)
            # for xid in range(len(adj_ball_x_frame)-1):
            #     if adj_ball_x_frame[xid+1] - adj_ball_x_frame[xid] > 20: continue
            #     ax.plot(adj_ball_x_frame[xid:xid+2], adj_ball_x_diff[xid:xid+2], marker='o', label="Adj", c='g', markersize=2, lw=0.5)
            # ax.set_xlim(-10, frame_count+10)
            # for hf in hit_frames:
            #     ax.axvline(x=hf, c='b', ls="--", lw=1)
            # ax.grid(axis='y')

            # ax = fig.add_subplot(3, 3, 7)
            # adj_ball_x, adj_ball_y = ball_df[["Adjusted X"]].values, ball_df[["Adjusted Y"]].values
            # pArwX = player_A_right_wrist_X = pose_df[["Player A right_wrist X"]].values
            # pArwY = player_A_right_wrist_Y = pose_df[["Player A right_wrist Y"]].values
            # pBrwX = player_B_right_wrist_X = pose_df[["Player B right_wrist X"]].values
            # pBrwY = player_B_right_wrist_Y = pose_df[["Player B right_wrist Y"]].values
            # diff_A  = (adj_ball_x-pArwX)**2 + (adj_ball_y-pArwY)**2
            # diff_A  = diff_A[(pArwX != 0) & (adj_ball_x != 0)]
            # frame_A = ball_df[["Frame"]][(pArwX != 0) & (adj_ball_x != 0)]
            # diff_B  = (adj_ball_x-pBrwX)**2 + (adj_ball_y-pBrwY)**2
            # diff_B  = diff_B[(pBrwX != 0) & (adj_ball_x != 0)]
            # frame_B = ball_df[["Frame"]][(pBrwX != 0) & (adj_ball_x != 0)]
            # ax.scatter(frame_A, diff_A, marker='o', label="A", c='g', s=3)
            # ax.scatter(frame_B, diff_B, marker='o', label="B", c='r', s=3)
            # ax.legend()
            # for hf in hit_frames: ax.axvline(x=hf, c='b', ls="--", lw=1)
            # ax.axhline(y=3000, c='b', ls="--", lw=1)
            # ax.set_xlim(-10, frame_count+10)
            # ax.set_ylim(110000, 1)
            # plt.yscale("log")
            # ax.grid(axis='y')

            # ax = fig.add_subplot(3, 3, 8)
            # ax.scatter(frame_A, diff_A, marker='o', label="A", c='g', s=3)
            # ax.legend()
            # for hf in hit_frames: ax.axvline(x=hf, c='b', ls="--", lw=1)
            # ax.axhline(y=3000, c='b', ls="--", lw=1)
            # ax.set_xlim(-10, frame_count+10)
            # ax.set_ylim(110000, 1)
            # plt.yscale("log")
            # ax.grid(axis='y')

            # ax = fig.add_subplot(3, 3, 9)
            # ax.scatter(frame_B, diff_B, marker='o', label="B", c='r', s=3)
            # ax.legend()
            # for hf in hit_frames:
            #     ax.axvline(x=hf, c='b', ls="--", lw=1)
            # ax.axhline(y=3000, c='b', ls="--", lw=1)
            # ax.set_xlim(-10, frame_count+10)
            # ax.set_ylim(110000, 1)
            # plt.yscale("log")
            # ax.grid(axis='y')

            # pArwX = player_A_right_wrist_X = pose_df[["Player A right_wrist X"]].values
            # pArwX_shift = np.concatenate([pArwX[1:], [pArwX[-1]]])
            # pArwY = player_A_right_wrist_Y = pose_df[["Player A right_wrist Y"]].values
            # pArwY_shift = np.concatenate([pArwY[1:], [pArwY[-1]]])
            # pBrwX = player_B_right_wrist_X = pose_df[["Player B right_wrist X"]].values
            # pBrwX_shift = np.concatenate([pBrwX[1:], [pBrwX[-1]]])
            # pBrwY = player_B_right_wrist_Y = pose_df[["Player B right_wrist Y"]].values
            # pBrwY_shift = np.concatenate([pBrwY[1:], [pBrwY[-1]]])
            # pArw_diff  = (pArwX_shift-pArwX)**2 + (pArwY_shift-pArwY)**2
            # pArw_frame = ball_df[["Frame"]][(pArwX_shift != 0) & (pArwX != 0) & (pArw_diff!=0)]
            # pArw_diff  = pArw_diff[(pArwX_shift != 0) & (pArwX != 0) & (pArw_diff!=0)]
            # pBrw_diff  = (pBrwX_shift-pBrwX)**2 + (pBrwY_shift-pBrwY)**2
            # pBrw_frame = ball_df[["Frame"]][(pBrwX_shift != 0) & (pBrwX != 0) & (pBrw_diff!=0)]
            # pBrw_diff  = pBrw_diff[(pBrwX_shift != 0) & (pBrwX != 0) & (pBrw_diff!=0)]

            # ax = fig.add_subplot(3, 3, 10)
            # ax.scatter(pArw_frame, pArw_diff, marker='o', label="A", c='g', s=3)
            # ax.scatter(pBrw_frame, pBrw_diff, marker='o', label="B", c='r', s=3)
            # ax.legend()
            # for hf in hit_frames: ax.axvline(x=hf, c='b', ls="--", lw=1)
            # # ax.axhline(y=3000, c='b', ls="--", lw=1)
            # ax.set_xlim(-10, frame_count+10)
            # # ax.set_ylim(110000, 1)
            # plt.yscale("log")
            # ax.grid(axis='y')

            # ax = fig.add_subplot(3, 3, 11)
            # ax.scatter(pArw_frame, pArw_diff, marker='o', label="A", c='g', s=3)
            # ax.legend()
            # for hf in hit_frames: ax.axvline(x=hf, c='b', ls="--", lw=1)
            # # ax.axhline(y=3000, c='b', ls="--", lw=1)
            # ax.set_xlim(-10, frame_count+10)
            # # ax.set_ylim(110000, 1)
            # plt.yscale("log")
            # ax.grid(axis='y')

            # ax = fig.add_subplot(3, 3, 12)
            # ax.scatter(pBrw_frame, pBrw_diff, marker='o', label="B", c='r', s=3)
            # ax.legend()
            # for hf in hit_frames: ax.axvline(x=hf, c='b', ls="--", lw=1)
            # # ax.axhline(y=3000, c='b', ls="--", lw=1)
            # ax.set_xlim(-10, frame_count+10)
            # # ax.set_ylim(110000, 1)
            # plt.yscale("log")
            # ax.grid(axis='y')

            # pArsX = player_A_right_shoulder_X = pose_df[["Player A right_shoulder X"]].values
            # pArsX_shift = np.concatenate([pArsX[1:], [pArsX[-1]]])
            # pArsY = player_A_right_shoulder_Y = pose_df[["Player A right_shoulder Y"]].values
            # pArsY_shift = np.concatenate([pArsY[1:], [pArsY[-1]]])
            # pBrsX = player_B_right_shoulder_X = pose_df[["Player B right_shoulder X"]].values
            # pBrsX_shift = np.concatenate([pBrsX[1:], [pBrsX[-1]]])
            # pBrsY = player_B_right_shoulder_Y = pose_df[["Player B right_shoulder Y"]].values
            # pBrsY_shift = np.concatenate([pBrsY[1:], [pBrsY[-1]]])
            # pArs_diff  = (pArsX_shift-pArsX)**2 + (pArsY_shift-pArsY)**2
            # pArs_frame = ball_df[["Frame"]][(pArsX_shift != 0) & (pArsX != 0) & (pArs_diff!=0)]
            # pArs_diff  = pArs_diff[(pArsX_shift != 0) & (pArsX != 0) & (pArs_diff!=0)]
            # pBrs_diff  = (pBrsX_shift-pBrsX)**2 + (pBrsY_shift-pBrsY)**2
            # pBrs_frame = ball_df[["Frame"]][(pBrsX_shift != 0) & (pBrsX != 0) & (pBrs_diff!=0)]
            # pBrs_diff  = pBrs_diff[(pBrsX_shift != 0) & (pBrsX != 0) & (pBrs_diff!=0)]

            # ax = fig.add_subplot(3, 3, 13)
            # ax.scatter(pArw_frame, pArw_diff-pArs_diff, marker='o', label="A", c='g', s=3)
            # ax.scatter(pBrw_frame, pBrw_diff-pBrs_diff, marker='o', label="B", c='r', s=3)
            # ax.legend()
            # for hf in hit_frames: ax.axvline(x=hf, c='b', ls="--", lw=1)
            # # ax.axhline(y=3000, c='b', ls="--", lw=1)
            # ax.set_xlim(-10, frame_count+10)
            # # ax.set_ylim(110000, 1)
            # # plt.yscale("log")
            # ax.grid(axis='y')

            # ax = fig.add_subplot(3, 3, 14)
            # ax.scatter(pArw_frame, pArw_diff-pArs_diff, marker='o', label="A", c='g', s=3)
            # ax.legend()
            # for hf in hit_frames: ax.axvline(x=hf, c='b', ls="--", lw=1)
            # # ax.axhline(y=3000, c='b', ls="--", lw=1)
            # ax.set_xlim(-10, frame_count+10)
            # # ax.set_ylim(110000, 1)
            # # plt.yscale("log")
            # ax.grid(axis='y')

            # ax = fig.add_subplot(3, 3, 15)
            # ax.scatter(pBrw_frame, pBrw_diff-pBrs_diff, marker='o', label="B", c='r', s=3)
            # ax.legend()
            # for hf in hit_frames: ax.axvline(x=hf, c='b', ls="--", lw=1)
            # # ax.axhline(y=3000, c='b', ls="--", lw=1)
            # ax.set_xlim(-10, frame_count+10)
            # # ax.set_ylim(110000, 1)
            # # plt.yscale("log")
            # ax.grid(axis='y')

            pass

        plt.suptitle(f"{video_id:05}.mp4 - HitFrame count: {len(hit_frames)}", fontsize=20)
        plt.tight_layout()
        plt.savefig(f"analysis/each_videos_trajectories/{video_id:05}.png")
        # plt.show()
        plt.close()
    return

def analysis_by_average_ratios():
    os.makedirs("analysis/average_ratios", exist_ok=True)
    for video_id in tqdm(train_formal_list):

        if video_id > 10: break

        hit_frames = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_S2.csv")["HitFrame"].values
        ball_df    = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_ball_33_adj.csv")
        adj_df     = ball_df.replace(np.nan, 0)  # [(ball_df["Adjusted X"] != 0) | (ball_df["Adjusted Y"] != 0)]
        adj_frames = adj_df[["Frame"]].values.squeeze()
        adj_ys     = 720 - adj_df[["Adjusted Y"]].values.squeeze()
        adj_xs     = adj_df[["Adjusted X"]].values.squeeze()

        adj_y_frames, adj_y_ratios = [], []
        adj_x_frames, adj_x_ratios = [], []
        for af in adj_frames:
            f_start = max(                0, af-5)
            f_end   = min(len(adj_frames)-1, af+5)

            xdata, ydata = adj_frames[f_start:f_end+1], adj_ys[f_start:f_end+1]
            xdata = xdata[ydata!=720]
            ydata = ydata[ydata!=720]
            assert len(xdata) == len(ydata)
            if len(xdata) > 1:
                popt, pconv = curve_fit(equation_1st, xdata, ydata)
                adj_y_frames.append(af)
                adj_y_ratios.append(popt[0])

            xdata, ydata = adj_frames[f_start:f_end+1], adj_xs[f_start:f_end+1]
            xdata = xdata[ydata!=0]
            ydata = ydata[ydata!=0]
            assert len(xdata) == len(ydata)
            if len(xdata) > 2:
                popt, pconv = curve_fit(equation_1st, xdata, ydata)
                adj_x_frames.append(af)
                adj_x_ratios.append(popt[0])


        fig = plt.figure(figsize=(20, 9))

        ax = fig.add_subplot(2, 2, 1)
        ax.scatter(adj_frames[adj_ys!=720], adj_ys[adj_ys!=720], label="Y", c='r', s=20)
        for hf in hit_frames: ax.axvline(x=hf, c='b', ls="--", lw=1)
        ax.legend()
        ax.grid(axis='y')

        ax = fig.add_subplot(2, 2, 3)
        ax.scatter(adj_y_frames, adj_y_ratios, label="Y", c='r', s=20)
        for hf in hit_frames: ax.axvline(x=hf, c='b', ls="--", lw=1)
        ax.legend()
        # ax.invert_yaxis()
        ax.grid(axis='y')

        ax = fig.add_subplot(2, 2, 2)
        ax.scatter(adj_frames[adj_xs!=0], adj_xs[adj_xs!=0], label="X", c='r', s=20)
        for hf in hit_frames: ax.axvline(x=hf, c='b', ls="--", lw=1)
        ax.legend()
        ax.grid(axis='y')
        
        ax = fig.add_subplot(2, 2, 4)
        ax.scatter(adj_x_frames, adj_x_ratios, label="X", c='r', s=20)
        for hf in hit_frames: ax.axvline(x=hf, c='b', ls="--", lw=1)
        ax.legend()
        ax.grid(axis='y')
        
        plt.suptitle(f"{video_id:05}.mp4 - HitFrame count: {len(hit_frames)}", fontsize=16)
        plt.tight_layout()
        plt.savefig(f"analysis/average_ratios/{video_id:05}.png")
        # plt.show()
        plt.close()

    return

def statisticize_hit_intervals():
    os.makedirs("analysis", exist_ok=True)
    fig = plt.figure(figsize=(20, 15))

    ax = fig.add_subplot(3, 1, 1)
    hit_intervals = []
    # for video_id in tqdm(range(1, 800+1)):
    for video_id in tqdm(train_formal_list):
        hit_frames = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_S2.csv")["HitFrame"].values
        hit_intervals += [ p-n for p, n in zip(hit_frames[1:-1], hit_frames[0:-2])]
    hit_inter_dict = { hi: hit_intervals.count(hi)
                       for hi in set(hit_intervals) }
    bar_container = plt.bar(hit_inter_dict.keys(), hit_inter_dict.values())
    ax.set_title("Hit intervals / amount", fontsize=16)
    plt.bar_label(bar_container, fmt="{:,.0f}")
    plt.xticks(np.arange(min(hit_inter_dict.keys()), max(hit_inter_dict.keys())+1, 1.0))
    plt.xlim(min(hit_inter_dict.keys())-1, max(hit_inter_dict.keys())+1)
    plt.grid()

    ax = fig.add_subplot(3, 1, 2)
    hit_counts = []
    # for video_id in tqdm(range(1, 800+1)):
    for video_id in tqdm(train_formal_list):
        hc = len(pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_S2.csv")["HitFrame"].values)
        hit_counts.append(hc)
    hit_count_dict = { hit: hit_counts.count(hit)
                       for hit in set(hit_counts) }
    bar_container = plt.bar(hit_count_dict.keys(), hit_count_dict.values())
    ax.set_title("Hit counts / Video amount", fontsize=16)
    plt.bar_label(bar_container, fmt="{:,.0f}")
    plt.xticks(np.arange(min(hit_count_dict.keys()), max(hit_count_dict.keys())+1, 1.0))
    plt.xlim(min(hit_count_dict.keys())-1, max(hit_count_dict.keys())+1)
    plt.grid()

    ax = fig.add_subplot(3, 1, 3)
    second_counts = []
    for video_id in tqdm(train_formal_list):
        sc = round(len(pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_pose.csv").values) / 30.0)
        second_counts.append(sc)
    bins = max(second_counts) - min(second_counts)
    plt.hist([ sc-0.5 for sc in second_counts ], bins=bins, lw=3, ec='w')
    ax.set_title("Second counts / Video amount", fontsize=16)
    # plt.bar_label(bar_container, fmt="{:,.0f}")
    for rect in ax.patches:
        height = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2, height+0.01, str(int(height)), ha="center", va="bottom")
    plt.xticks(np.arange(min(second_counts), max(second_counts), 1.0))
    plt.xlim(min(second_counts)-1, max(second_counts))
    plt.grid()

    # plt.suptitle("Statistic of Hit Intervals", fontsize=16)
    plt.tight_layout()
    plt.savefig("analysis/hit_intervals.png")
    # plt.show()
    return

def merge_close_frames(frames:"np.ndarray[int]"):
    frames:list[int] = list(frames)
    fid = 0
    frames_to_avg = []
    frames_to_append = []
    start_flag = False
    while True:
        if fid==len(frames)-1: break
        if frames[fid+1]-frames[fid] <= 7:
            frames_to_avg.append(frames.pop(fid))
            start_flag = True
        elif start_flag:
            frames_to_avg.append(frames.pop(fid))
            assert len(frames_to_avg) >= 2
            avgf = int(round(sum(frames_to_avg)/len(frames_to_avg)))
            frames_to_append.append(avgf)
            frames_to_avg = []
            start_flag = False
        else:
            fid += 1
    frames += frames_to_append
    return np.array(sorted(frames))

def statisticize_findpeaks_accuracy():
    # os.makedirs("analysis/findpeaks_accuracy", exist_ok=True)
    pcccs, ncccs, acccs, tcccs = [], [], [], []
    for video_id in tqdm(train_formal_list):

        if video_id > 5: break

        hit_frames  = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_S2.csv")["HitFrame"].values
        ball_df     = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_ball_33_adj.csv")

        adj_ball_df    = ball_df.replace(np.nan, 0)
        adj_ball_df    = adj_ball_df[(adj_ball_df["Adjusted X"]!=0) | (adj_ball_df["Adjusted Y"]!=0)]
        adj_ball_frame = adj_ball_df[["Frame"]].values.squeeze()
        adj_ball_x     = adj_ball_df[["Adjusted X"]].values.squeeze()
        adj_ball_y     = adj_ball_df[["Adjusted Y"]].values.squeeze()

        pabxp = sorted(find_peaks(adj_ball_x     , distance=7, prominence=12)[0])  # positive_adj_ball_x_peaks
        nabxp = sorted(find_peaks(adj_ball_x*(-1), distance=7, prominence=12)[0])  # negative_adj_ball_x_peaks
        abxp  = sorted(list(set(list(pabxp) + list(nabxp))))                       #          adj_ball_x_peaks
        abyp  = sorted(find_peaks(adj_ball_y     , distance=7, prominence=12)[0])  #          adj_ball_y_peaks
        abtp  = sorted(list(set(list(abxp) + list(abyp))))                         #    total_adj_ball_peaks

        abxpf  = adj_ball_frame[abxp]  # adj_ball_x_peaks_frame
        abypf  = adj_ball_frame[abyp]  # adj_ball_y_peaks_frame
        abtpf  = adj_ball_frame[abtp]  # adj_ball_t_peaks_frame (t for total)

        pccc = len(pabxp) == len(hit_frames)  # pabxp_count_completely_correct
        nccc = len(nabxp) == len(hit_frames)  # nabxp_count_completely_correct
        accc = len( abyp) == len(hit_frames)  #  abyp_count_completely_correct
        tccc = len( abtp) == len(hit_frames)  #  abtp_count_completely_correct

        pcccs.append(pccc)
        ncccs.append(nccc)
        acccs.append(accc)
        tcccs.append(tccc)

        print(f"\n{video_id:05}\n",
              abtpf, '\n',
              merge_close_frames(abtpf), '\n',
            #   abxpf, '\n',
            #   abypf, '\n',
            #   abtpf, '\n',
              hit_frames)
    
    return

# def statisticize_coeffs():
#     fig = plt.figure(figsize=(20, 20))
#     for video_id in tqdm(train_formal_list):
#         hit_frames  = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_S2.csv")["HitFrame"].values
#         ball_df     = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_ball_33_adj.csv")
#         if len(hit_frames)
#         for hf in hit_frames[]

def preview_curves_fitting():
    video_id = 3
    hit_frames = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_S2.csv")["HitFrame"].values
    ball_df = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_ball_33_adj.csv")
    adj_ball_frame = ball_df[["Frame"]].values
    adj_ball_y     = ball_df[["Adjusted Y"]].values

    plot_frame = adj_ball_frame[adj_ball_y!=0]
    plot_y     = adj_ball_y[adj_ball_y!=0]
    plot_y     = 720 - plot_y
    plt.figure(figsize=(20, 5))
    plt.scatter(plot_frame, plot_y, s=3, c='b')

    for hit_id in range(len(hit_frames)):
        hf_start = hit_frames[hit_id]
        if hit_id == len(hit_frames)-1:
            frame_section    = adj_ball_frame[hf_start:-1]
            aby_section      = adj_ball_y[hf_start:-1]
        else:
            hf_end = hit_frames[hit_id+1]
            frame_section    = adj_ball_frame[hf_start:hf_end]
            aby_section      = adj_ball_y[hf_start:hf_end]
        frame_section    = frame_section[aby_section!=0]
        aby_section      = aby_section[aby_section!=0]
        aby_section      = 720 - aby_section
        bounds_3rd = [ ( -np.inf, -10, -np.inf, -np.inf ),
                       (  np.inf,   0,  np.inf,  np.inf ) ]
        bounds_2nd = [ ( -np.inf, -np.inf, -np.inf ),
                       (       0,  np.inf,  np.inf ) ]
        popt, pcov = curve_fit(equation_2nd, frame_section, aby_section, bounds=bounds_2nd)
        x_line = np.arange(min(frame_section)-10, max(frame_section)+10, 1)
        y_line = equation_2nd(x_line, *popt)
        y_mse  = ((aby_section-equation_2nd(frame_section, *popt))**2).sum() / len(frame_section)
        print(*popt, y_mse)
        plt.plot(x_line, y_line, color='r', lw=1)

    plt.suptitle("curves fitting", fontsize=16)
    plt.tight_layout()
    # plt.savefig("output/trajectory/curves_fitting.png")
    plt.show()
    plt.close()
    return

def preview_fft():
    video_id = 3
    hit_count      = len(pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_S2.csv")["HitFrame"].values)
    ball_df        = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_ball_33_adj.csv")
    adj_ball_frame = ball_df[["Frame"]]
    adj_ball_y     = ball_df[["Adjusted Y"]]

    plot_y     = adj_ball_y.replace(np.nan, 0).values
    plot_frame = adj_ball_frame.values[plot_y != 0]
    plot_y     = 720 - plot_y[plot_y != 0]
    plt.figure(figsize=(20, 5))
    plt.scatter(plot_frame, plot_y, s=3, c='b')

    popt, pcov = curve_fit(fourier, plot_frame, plot_y, [1.0]*50)
    x_line = np.arange(min(plot_frame)-10, max(plot_frame)+10, 1)
    y_line = fourier(x_line, *popt)
    y_mse  = ((plot_y-fourier(plot_frame, *popt))**2).sum() / len(plot_frame)
    print(*popt, y_mse)
    plt.plot(x_line, y_line, color='r', lw=1)
    plt.suptitle(f"{video_id:05}.mp4 FFT - hit count: {hit_count}", fontsize=16)
    plt.tight_layout()
    # plt.savefig("output/trajectory/curves_fitting.png")
    plt.show()
    plt.close()
    return

def curves_fitting():
    os.makedirs("output/curves_fitting", exist_ok=True)
    video_id = 3
    hit_frames        = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_S2.csv")["HitFrame"].values
    ball_df           = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_ball_33_adj.csv")
    adj_ball_frame    = ball_df[["Frame"]].values
    adj_ball_y        = ball_df[["Adjusted Y"]].replace(np.nan, 0).values
    adj_ball_frame    = adj_ball_frame[adj_ball_y!=0]
    adj_ball_y        = adj_ball_y[adj_ball_y!=0]
    frame_count       = max(adj_ball_frame)
    peaks, properties = find_peaks(adj_ball_y, distance=7, prominence=12)
    adj_ball_y        = 720 - adj_ball_y

    fig = plt.figure(figsize=(16, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(adj_ball_frame[peaks], adj_ball_y[peaks], label="Peak", c='r', s=20)
    ax.scatter(adj_ball_frame, adj_ball_y, s=3, label="Adj", c='g')
    ax.set_xlim(-10, frame_count+10)
    for hf in hit_frames: ax.axvline(x=hf, c='b', ls="--", lw=1)
    ax.legend()
    ax.grid(axis='y')
    plt.suptitle(f"{video_id:05}.mp4 - HitFrame count: {len(hit_frames)} / Peak count: {len(peaks)}", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"output/curves_fitting/{video_id:05}.png")
    plt.show()
    # plt.close()
    return


if __name__ == "__main__":
    # plot_each_videos_trajectories()
    # analysis_by_average_ratios()
    # statisticize_hit_intervals()
    # statisticize_findpeaks_accuracy()
    statisticize_coeffs()
    # preview_curves_fitting()
    # preview_fft()
    # curves_fitting()
    pass