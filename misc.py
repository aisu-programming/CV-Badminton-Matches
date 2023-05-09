import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

train_formal_list = [
    1, 2, 3, 4, 5, 8, 9, 10, 11, 14, 17, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 38, 39, 40,
    41, 42, 45, 46, 47, 48, 49, 50, 52, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 68, 69, 70, 71,
    72, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 93, 94, 95, 96, 97, 98, 99,
    101, 102, 103, 104, 105, 107, 108, 109, 110, 111, 112, 113, 114, 115, 117, 118, 119, 120, 121, 122, 123,
    125, 126, 127, 128, 129, 130, 131, 132, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 146, 147,
    148, 149, 150, 151, 153, 154, 155, 156, 157, 158, 159, 160, 161, 163, 165, 166, 167, 168, 169, 170, 172,
    173, 174, 176, 177, 178, 179, 180, 181, 182, 184, 185, 189, 190, 191, 192, 194, 195, 197, 198, 199, 200,
    201, 202, 203, 204, 205, 206, 207, 208, 209, 211, 212, 213, 214, 215, 216, 217, 218, 221, 222, 223, 224,
    225, 226, 227, 228, 229, 231, 232, 234, 235, 236, 238, 239, 240, 241, 242, 243, 244, 246, 247, 249, 253,
    254, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275,
    276, 277, 278, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 292, 293, 295, 296, 297, 298, 301,
    302, 303, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 316, 317, 319, 320, 321, 322, 323, 325, 327,
    328, 330, 331, 332, 333, 335, 336, 337, 340, 341, 343, 344, 345, 346, 347, 348, 350, 351, 352, 353, 355,
    356, 357, 358, 361, 362, 363, 364, 365, 366, 367, 368, 369, 371, 372, 373, 374, 375, 376, 377, 378, 379,
    380, 381, 382, 383, 384, 385, 386, 387, 388, 390, 391, 392, 393, 394, 395, 396, 398, 399, 400, 401, 402,
    403, 405, 406, 407, 408, 409, 410, 411, 412, 413, 415, 416, 417, 418, 419, 421, 422, 423, 424, 425, 426,
    427, 428, 429, 431, 432, 435, 436, 437, 438, 440, 442, 443, 445, 446, 447, 449, 452, 453, 454, 456, 457,
    459, 461, 462, 464, 467, 468, 470, 473, 474, 475, 476, 477, 478, 479, 481, 482, 483, 484, 485, 486, 487,
    488, 489, 490, 491, 492, 494, 495, 496, 497, 498, 499, 500, 502, 503, 505, 507, 510, 511, 512, 513, 515,
    516, 517, 519, 521, 522, 524, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540,
    541, 542, 543, 544, 545, 547, 548, 549, 550, 551, 552, 553, 556, 557, 558, 559, 561, 562, 563, 564, 565,
    566, 567, 570, 571, 572, 573, 574, 575, 577, 580, 581, 582, 583, 584, 585, 587, 588, 589, 593, 594, 595,
    598, 599, 600, 601, 602, 603, 604, 605, 606, 608, 609, 610, 611, 613, 614, 615, 617, 618, 619, 620, 622,
    623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 634, 635, 636, 637, 638, 639, 641, 642, 644, 645, 646,
    647, 648, 649, 650, 651, 653, 654, 655, 657, 658, 660, 662, 663, 664, 666, 667, 668, 669, 670, 672, 673,
    674, 675, 676, 678, 679, 680, 681, 682, 683, 684, 685, 686, 688, 689, 690, 691, 692, 693, 694, 695, 697,
    698, 699, 700, 701, 703, 704, 706, 707, 709, 711, 712, 713, 715, 716, 717, 718, 720, 721, 722, 723, 724,
    725, 726, 727, 728, 731, 732, 733, 734, 735, 736, 737, 739, 740, 741, 744, 745, 746, 747, 748, 749, 750,
    751, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 769, 770, 771, 772, 773,
    774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 791, 792, 794, 795, 796, 797, 798, 799, 800
]

train_informal_list = [
    6, 7, 12, 13, 15, 16, 18, 19, 20, 21, 36, 37, 43, 44, 51, 53, 66, 73, 92, 100, 106,116, 124, 133, 145,
    152, 162, 164, 171, 175, 183, 186, 187, 188, 193, 196, 210, 219, 220, 230, 233, 237, 245, 248, 250, 251,
    252, 255, 279, 291, 294, 299, 300, 304, 315, 318, 324, 326, 329, 334, 338, 339, 342, 349, 354, 359, 360,
    370, 389, 397, 404, 414, 420, 430, 433, 434, 439, 441, 444, 448, 450, 451, 455, 458, 460, 463, 465, 466,
    469, 471, 472, 480, 493, 501, 504, 506, 508, 509, 514, 518, 520, 523, 525, 546, 554, 555, 560, 568, 569,
    576, 578, 579, 586, 590, 591, 592, 596, 597, 607, 612, 616, 621, 633, 640, 643, 652, 656, 659, 661, 665,
    671, 677, 687, 696, 702, 705, 708, 710, 714, 719, 729, 730, 738, 742, 743, 752, 768, 790, 793
]

valid_formal_list = [
    1, 3, 4, 5, 6, 7, 9, 11, 12, 13, 14, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 52, 53, 54, 55, 57, 58, 59, 60, 61, 62,
    63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 76, 77, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 90, 91, 93, 94,
    95, 96, 97, 98, 99, 100, 101, 102, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 115, 116, 117, 118,
    119, 123, 124, 125, 127, 128, 129, 130, 131, 132, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144,
    145, 146, 147, 148, 149, 150, 151, 152, 153, 155, 157, 158, 159, 161, 162, 163, 164, 166, 167, 169
]

valid_informal_list = [
    2, 8, 10, 15, 18, 27, 51, 56, 65, 71, 75, 78, 89, 92, 103, 114, 120, 121, 122, 126, 133, 154, 156, 160, 165, 168
]

def get_video_id_list():
    formal_mp4s = os.listdir("data/train_mp4/informal")
    print([ int(filename.split('.')[0]) for filename in formal_mp4s ])

def view_video_infos():
    import mmcv
    for video_filename in os.listdir("data/train_mp4/informal"):
        video = mmcv.VideoReader(f"data/train_mp4/informal/{video_filename}")
        if video.width != 1280 or video.height != 720:
            print(video_filename, video.width, video.height)

def plot_first_frames():
    import mmcv
    amount = 7
    plt.figure(figsize=(16*amount, 9*amount))
    for video_id in range(1, amount*amount+1):
        plt.subplot(amount, amount, video_id)
        img = mmcv.VideoReader(f"data/train/{video_id:05}/{video_id:05}.mp4")[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.axis("off")
    plt.tight_layout()
    plt.show()
    plt.savefig(f"head_{amount*amount}_first_frame", dpi=500)
    plt.close()

def darken_video(video_id):
    import mmcv
    videoReader = mmcv.VideoReader(f"data/train/{video_id:05}/{video_id:05}.mp4")
    width, height = videoReader.width, videoReader.height
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = videoReader.fps
    size = (width, height)
    videoWriter = cv2.VideoWriter(f"data/train/{video_id:05}/{video_id:05}_darken.mp4", fourcc, fps, size)
    video = videoReader[:]
    for frame in video:
        frame = np.uint8(np.minimum((frame/255*1.3), 1.0) **2 *255)
        videoWriter.write(frame)
    videoWriter.release()
    return

def crop_video(video_id):
    import mmcv
    videoReader = mmcv.VideoReader(f"data/train/{video_id:05}/{video_id:05}.mp4")
    width, height = videoReader.width, videoReader.height
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = videoReader.fps
    size = (width, height)
    videoWriter = cv2.VideoWriter(f"data/train/{video_id:05}/{video_id:05}_crop.mp4", fourcc, fps, size)
    video = videoReader[:-17]
    # for fid, frame in enumerate(video):
    #     # cv2.imshow(str(fid), frame)
    #     # cv2.waitKey(0)
    #     videoWriter.write(frame)
    # videoWriter.release()
    for fid, frame in enumerate(reversed(video)):
        cv2.imshow(str(fid), frame)
        cv2.waitKey(0)
        cv2.destroyWindow(str(fid))
        videoWriter.write(frame)
    videoWriter.release()
    return

def plot_image():
    image = cv2.imread("data/train_background/00011.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()
    return

def plot_background_lines():
    from data.train_background.classification import background_details
    for bg_id in range(12):
        image = cv2.imread(f"data/train_background/{bg_id:05}.png")
        bgd = background_details[bg_id]
        left_top     = (bgd["x_left_top"],     bgd["y_top"])
        right_top    = (bgd["x_right_top"],    bgd["y_top"])
        left_bottom  = (bgd["x_left_bottom"],  bgd["y_bottom"])
        right_bottom = (bgd["x_right_bottom"], bgd["y_bottom"])
        points = [ left_top, left_bottom, right_bottom, right_top, left_top ]
        for pid in range(4):
            image = cv2.line(image, points[pid], points[pid+1], (128,255,255), 3)
        cv2.imshow(str(bg_id), image)
        cv2.waitKey(0)
        cv2.destroyWindow(str(bg_id))
    return

def check_length_match():
    for video_id in tqdm(train_formal_list):
        video_ori_frame_count  = cv2.VideoCapture(f"data/train/{video_id:05}/{video_id:05}.mp4").get(cv2.CAP_PROP_FRAME_COUNT)
        video_pose_frame_count = cv2.VideoCapture(f"data/train/{video_id:05}/{video_id:05}_pose.mp4").get(cv2.CAP_PROP_FRAME_COUNT)
        if not (video_ori_frame_count == video_pose_frame_count):
            print(video_id, video_ori_frame_count, video_pose_frame_count)
    return

def combine_ball_and_pose_csv_v1():
    for video_id in tqdm(train_formal_list):
        pose_csv = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_pose.csv")
        ball_csv = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_ball_33_adj.csv")
        # pose_csv = pose_csv.drop([ "Frame", "Player A confidence", "Player B confidence" ], axis=1)
        pose_csv = pose_csv.drop([ "Frame" ], axis=1)
        pose_csv_columns = pose_csv.columns
        for column in pose_csv_columns:
            if   " X" in column: pose_csv[[column]] = pose_csv[[column]] / 1280
            elif " Y" in column: pose_csv[[column]] = pose_csv[[column]] / 720
        ball_csv = ball_csv[[ "Adjusted Vis", "Adjusted X", "Adjusted Y" ]]
        ball_csv[["Adjusted X"]] = ball_csv[["Adjusted X"]] / 1280
        ball_csv[["Adjusted Y"]] = ball_csv[["Adjusted Y"]] / 720
        combined_csv = pd.concat([ pose_csv, ball_csv ], axis=1)
        combined_csv.index.set_names("Frame", inplace=True)
        combined_csv = combined_csv.fillna(0)
        combined_csv.to_csv(f"data/train/{video_id:05}/{video_id:05}_combined.csv")
    return

def convert_ground_truth_v1():
    # from scipy.ndimage import gaussian_filter
    for video_id in tqdm(train_formal_list):
        frame_count = int(cv2.VideoCapture(f"data/train/{video_id:05}/{video_id:05}.mp4").get(cv2.CAP_PROP_FRAME_COUNT))
        hit_data    = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_S2.csv")[["HitFrame", "Hitter"]].values
        hit_A = np.zeros(frame_count, dtype=np.uint8)
        hit_B = np.zeros(frame_count, dtype=np.uint8)
        flag = None
        for hf, htr in hit_data:
            if flag is None:
                flag, last_hf = htr, hf
            elif flag == 'A':
                hit_A[last_hf:hf] = 1
                flag, last_hf = htr, hf
            elif flag == 'B':
                hit_B[last_hf:hf] = 1
                flag, last_hf = htr, hf
            else:
                raise Exception
        if   flag == 'A': hit_A[last_hf:] = 1
        elif flag == 'B': hit_B[last_hf:] = 1
        hit_B = hit_B * 2

        ground_truth = pd.DataFrame({
            "Frame" : pd.Series(range(frame_count)),
            "Hitter": pd.Series(hit_A+hit_B),
        })
        ground_truth = ground_truth.set_index("Frame")
        ground_truth.to_csv(f"data/train/{video_id:05}/{video_id:05}_S2_hit.csv")
    return

def split_video_into_images():
    import mmcv
    for video_id in tqdm(train_formal_list):
        if video_id > 1: break
        os.makedirs(f"data/train/{video_id:05}/images", exist_ok=True)
        video = mmcv.VideoReader(f"data/train/{video_id:05}/{video_id:05}.mp4")[:]
        for img_id, img in enumerate(video):
            cv2.imwrite(f"data/train/{video_id:05}/images/{img_id:04}.jpg", img)
    return

def load_images():
    videos = []
    for video_id in tqdm(train_formal_list):
        images = []
        images_count = len(os.listdir(f"data/train/{video_id:05}/images"))
        for img_id in range(images_count):
            img = cv2.imread(f"data/train/{video_id:05}/images/{img_id:04}.jpg")
            images.append(img)
            # with open(f"data/train/{video_id:05}/images/{img_id:04}.jpg", mode="rb") as i: 
            #     img = i.read()
            #     images.append(img)
        videos.append(images)
    return

def create_h5py():
    import h5py
    with h5py.File("data/train/images.hdf5", "w") as h5py_file:
        for video_id in tqdm(train_formal_list):
            images = []
            images_count = len(os.listdir(f"data/train/{video_id:05}/images"))
            for img_id in range(images_count):
                # img = cv2.imread(f"data/train/{video_id:05}/images/{img_id:04}.jpg")
                # images.append(img)
                with open(f"data/train/{video_id:05}/images/{img_id:04}.jpg", 'rb') as img:
                    images.append(img.read())
            images = np.asarray(images)
            h5py_file.create_dataset(f"{video_id:05}", data=images)
    return

def scale_images():
    import mmcv
    for video_id in tqdm(train_formal_list):
        if video_id < 135: continue
        os.makedirs(f"data/train/{video_id:05}/images_0.5", exist_ok=True)
        video = mmcv.VideoReader(f"data/train/{video_id:05}/{video_id:05}.mp4")[:]
        for img_id, img in enumerate(video):
            img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(f"data/train/{video_id:05}/images_0.5/{img_id:04}.jpg", img)
    return

def analyze_hits():
    # min_frame_diffs = []
    # for video_id in tqdm(train_formal_list):
    #     frame_diffs = []
    #     hit_frame   = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_S2.csv")[["HitFrame"]].values.flatten()
    #     frame_count = int(cv2.VideoCapture(f"data/train/{video_id:05}/{video_id:05}.mp4").get(cv2.CAP_PROP_FRAME_COUNT))
    #     last_frame = 0
    #     for hid in range(len(hit_frame)+1):
    #         if hid != len(hit_frame):
    #             fd = hit_frame[hid] - last_frame
    #             frame_diffs.append(fd)
    #             last_frame = hit_frame[hid]
    #         else:
    #             fd = frame_count - last_frame
    #             frame_diffs.append(fd)
    #     frame_diffs = sorted(frame_diffs)[:3]
    #     min_frame_diffs.append(frame_diffs)
    # min_frame_diffs = sorted(min_frame_diffs, key=lambda fds: fds[0])
    # print(min_frame_diffs[:10])
    frame_diffs = []
    for video_id in tqdm(train_formal_list):
        frame_count = int(cv2.VideoCapture(f"data/train/{video_id:05}/{video_id:05}.mp4").get(cv2.CAP_PROP_FRAME_COUNT))
        hit_frame   = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_S2.csv")[["HitFrame"]].values.flatten()
        frame_diffs.append(frame_count-hit_frame[-1])
    print(sorted(frame_diffs))
    return

def scale_and_crop_images():
    os.makedirs("data/train_background/images_0.25", exist_ok=True)
    for bg_filename in os.listdir("data/train_background"):
        if ".png" in bg_filename:
            img = cv2.imread(f"data/train_background/{bg_filename}")
            img = cv2.resize(img, (320, 180), interpolation=cv2.INTER_CUBIC)
            img = cv2.copyMakeBorder(img, 35, 35, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
            img = img[:, 35:-35]
            cv2.imwrite(f"data/train_background/images_0.25/{bg_filename}", img)
    # import mmcv
    # os.makedirs(f"data/train/{video_id:05}/images_0.25", exist_ok=True)
    # for video_id in tqdm(train_formal_list):
    #     video = mmcv.VideoReader(f"data/train/{video_id:05}/{video_id:05}.mp4")[:]
    #     for img_id, img in enumerate(video):
    #         img = cv2.resize(img, (320, 180), interpolation=cv2.INTER_CUBIC)
    #         img = cv2.copyMakeBorder(img, 35, 35, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
    #         img = img[:, 35:-35]
    #         cv2.imwrite(f"data/train/{video_id:05}/images_0.25/{img_id:04}.jpg", img)
    return

def convert_ground_truth_v2():
    for video_id in tqdm(train_formal_list):
        frame_count = int(cv2.VideoCapture(f"data/train/{video_id:05}/{video_id:05}.mp4").get(cv2.CAP_PROP_FRAME_COUNT))
        hit_data    = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_S2.csv")[["HitFrame", "Hitter"]].values
        hit_A = np.zeros(frame_count, dtype=np.float32)
        hit_B = np.zeros(frame_count, dtype=np.float32)
        for hf, htr in hit_data:
            if htr == 'A': hit_A[max(0, hf-1):hf+1] = 1.0
            else:          hit_B[max(0, hf-1):hf+1] = 1.0
        hit = np.minimum(1.0, hit_A+hit_B)
        ground_truth = pd.DataFrame({
            "Frame": pd.Series(range(frame_count)),
            "Hit"  : pd.Series(hit),
            "Hit A": pd.Series(hit_A),
            "Hit B": pd.Series(hit_B),
        })
        ground_truth = ground_truth.set_index("Frame")
        ground_truth.to_csv(f"data/train/{video_id:05}/{video_id:05}_S2_hit.csv")
    return

def crop_images():
    import mmcv
    for video_id in tqdm(train_formal_list):
        os.makedirs(f"data/train/{video_id:05}/square", exist_ok=True)
        video = mmcv.VideoReader(f"data/train/{video_id:05}/{video_id:05}.mp4")[:]
        for img_id, img in enumerate(video):
            img = cv2.copyMakeBorder(img, 140, 140, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
            img = img[:, 140:-140]
            cv2.imwrite(f"data/train/{video_id:05}/square/{img_id:04}.jpg", img)
    return

def get_max_frame_count():
    max_frame_count = 0
    for video_id in tqdm(range(1, 800+1)):
        frame_count = int(cv2.VideoCapture(f"data/train/{video_id:05}/{video_id:05}.mp4").get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count > max_frame_count: max_frame_count = frame_count
    for video_id in tqdm(range(1, 169+1)):
        frame_count = int(cv2.VideoCapture(f"data/val/{video_id:05}/{video_id:05}.mp4").get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count > max_frame_count: max_frame_count = frame_count
    print(max_frame_count)
    return

def get_max_hit_count():
    max_hit_count = 0
    for video_id in tqdm(range(1, 800+1)):
        hit_count = len(pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_S2.csv")[["Hitter"]].values)
        if hit_count > max_hit_count: max_hit_count = hit_count
    print(max_hit_count)
    return

def get_hitter():
    for video_id in tqdm(range(1, 800+1)):
        hitter = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_S2.csv")[["Hitter"]].values
        if hitter[0, 0] != 'B': print(video_id)
    return

def combine_ball_and_pose_csv_v2():
    for video_id in tqdm(train_formal_list):
        pose_df = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_pose.csv")
        pose_df = pose_df[[
            "Player A right_wrist X", "Player A right_wrist Y",
            "Player B right_wrist X", "Player B right_wrist Y"
        ]]
        for column in pose_df.columns:
            if   " X" in column: pose_df[[column]] = pose_df[[column]] / 1280
            elif " Y" in column: pose_df[[column]] = pose_df[[column]] / 720

        ball_df = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_ball_33_adj.csv")
        ball_df = ball_df[[ "Adjusted Vis", "Adjusted X", "Adjusted Y" ]]
        ball_df[["Adjusted X"]] = ball_df[["Adjusted X"]] / 1280
        ball_df[["Adjusted Y"]] = ball_df[["Adjusted Y"]] / 720

        pArwX = pose_df[["Player A right_wrist X"]].values.copy().squeeze()
        adj_ball_x = ball_df[["Adjusted X"]].values.copy().squeeze()
        pArwX[adj_ball_x==0] = 0
        adj_ball_x[pArwX==0] = 0
        pArwY = pose_df[["Player A right_wrist Y"]].values.copy().squeeze()
        adj_ball_y = ball_df[["Adjusted Y"]].values.copy().squeeze()
        pArwY[adj_ball_y==0] = 0
        adj_ball_y[pArwY==0] = 0
        pArw_ball_distance = (adj_ball_x-pArwX)**2 + (adj_ball_y-pArwY)**2

        pBrwX = pose_df[["Player B right_wrist X"]].values.copy().squeeze()
        adj_ball_x = ball_df[["Adjusted X"]].values.copy().squeeze()
        pBrwX[adj_ball_x==0] = 0
        adj_ball_x[pBrwX==0] = 0
        pBrwY = pose_df[["Player B right_wrist Y"]].values.copy().squeeze()
        adj_ball_y = ball_df[["Adjusted Y"]].values.copy().squeeze()
        pBrwY[adj_ball_y==0] = 0
        adj_ball_y[pBrwY==0] = 0
        pBrw_ball_distance = (adj_ball_x-pBrwX)**2 + (adj_ball_y-pBrwY)**2

        adj_ball_x = ball_df[["Adjusted X"]].values.copy().squeeze()
        adj_ball_x_shift = np.concatenate([adj_ball_x[1:], adj_ball_x[-1:]])
        adj_ball_x[adj_ball_x_shift==0] = 0
        adj_ball_x_shift[adj_ball_x==0] = 0
        adj_ball_x_diff = adj_ball_x_shift - adj_ball_x
        adj_ball_x_diff = adj_ball_x_diff

        adj_ball_y = ball_df[["Adjusted Y"]].values.copy().squeeze()
        adj_ball_y_shift = np.concatenate([adj_ball_y[1:], adj_ball_y[-1:]])
        adj_ball_y[adj_ball_y_shift==0] = 0
        adj_ball_y_shift[adj_ball_y==0] = 0
        adj_ball_y_diff = adj_ball_y_shift - adj_ball_y
        adj_ball_y_diff = adj_ball_y_diff

        adj_ball_diff = (adj_ball_x_diff**2 + adj_ball_y_diff**2)**0.5
        ball_df_add = pd.DataFrame({
            "A dis" : pArw_ball_distance,
            "B dis" : pBrw_ball_distance,
            "X diff": adj_ball_x_diff,
            "Y diff": adj_ball_y_diff,
            "diff"  : adj_ball_diff,
        })

        combined_df = pd.concat([ ball_df, ball_df_add ], axis=1)
        combined_df.index.set_names("Frame", inplace=True)
        combined_df = combined_df.replace(0, np.nan)
        for column in [
            "Adjusted X", "Adjusted Y",
            "A dis", "B dis",
            "X diff", "Y diff", "diff",
        ]:
            col_data = combined_df[column].copy()
            col_data = (col_data-col_data.min())/(col_data.max()-col_data.min())
            col_data = (col_data-0.5) *2
            combined_df[column] = col_data
        combined_df = combined_df.fillna(0)
        combined_df.to_csv(f"data/train/{video_id:05}/{video_id:05}_combined.csv")
    return

def detect_same_frames():  # Failed
    video_id = 1
    img1 = cv2.imread(f"data/train/{video_id:05}/images/0024.jpg")  # , cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(f"data/train/{video_id:05}/images/0025.jpg")  # , cv2.IMREAD_GRAYSCALE)
    img3 = cv2.imread(f"data/train/{video_id:05}/images/0026.jpg")  # , cv2.IMREAD_GRAYSCALE)
    img4 = abs(img1-img2)
    img5 = abs(img2-img3)
    print(img1.shape, img2.shape, img4.shape, img4)
    print(img5.sum())
    return


def create_posec3d_dataset():
    import mmcv, random, mmengine
    
    split = { "xsub_train": [], "xsub_val": [] }
    annotations = []
    for video_id in tqdm(train_formal_list):

        if video_id > 10: break
        
        hit_data  = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_S2.csv")[["HitFrame", "Hitter", "BallType"]].values
        pose_data = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_pose.csv").replace(np.nan, 0).values
        
        video_reader = mmcv.VideoReader(f"data/train/{video_id:05}/{video_id:05}.mp4")
        width, height = video_reader.width, video_reader.height
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = video_reader.fps
        size = (width, height)
        video = video_reader[:]

        os.makedirs("dataset_poseC3D/videos", exist_ok=True)
        for hid, (hit_frame, hitter, ball_type) in enumerate(hit_data):
            hf_start, hf_end = max(hit_frame-8, 0), min(hit_frame+8, len(pose_data))
            if hitter == 'B': ball_type += 9

            video_filename = f"dataset_poseC3D/videos/{video_id:05}_{hid}.mp4"
            # video_writer = cv2.VideoWriter(video_filename, fourcc, fps, size)
            # for hf in range(hf_start, hf_end+1): video_writer.write(video[hf])
            # video_writer.release()

            total_keypoints, total_keypoint_scores = [], []
            for hf in range(hf_start, hf_end+1):
                keypoints, keypoint_scores = [], []
                if pose_data[hf, 1] != 0:
                    kpt = np.dstack([pose_data[hf,  6:( 6+34):2], pose_data[hf, ( 6+1):( 6+34):2]]).squeeze()
                    assert kpt.shape == (17, 2)
                    keypoints.append(kpt.tolist())
                    keypoint_scores.append([pose_data[hf, 1].tolist()] * 17)
                if pose_data[hf, 40] != 0:
                    kpt = np.dstack([pose_data[hf, 45:(45+34):2], pose_data[hf, (45+1):(45+34):2]]).squeeze()
                    assert kpt.shape == (17, 2)
                    keypoints.append(kpt.tolist())
                    keypoint_scores.append([pose_data[hf, 40].tolist()] * 17)
                total_keypoints.append(keypoints)
                total_keypoint_scores.append(keypoint_scores)
            
            # if random.random() < 0.8:
            #     split["xsub_train"].append(video_filename)
            # else:
            #     split["xsub_val"].append(video_filename)
            # annotations.append()

            if random.random() < 0.8:
                pkl_filename = "dataset_poseC3D/custom_dataset_train.pkl"
            else:
                pkl_filename = "dataset_poseC3D/custom_dataset_val.pkl"
            mmengine.dump({
                        "frame_dir"     : f"videos/{video_id:05}_{hid}.mp4",
                        "label"         : int(ball_type),
                        "img_shape"     : size,
                        "original_shape": size,
                        "total_frames"  : int(hf_end - hf_start),
                        "keypoint"      : total_keypoints,
                        "keypoint_score": total_keypoint_scores,
                    }, pkl_filename)

    # with open("dataset_poseC3D/annotations.pkl", 'w') as pkl_file:
    #     json.dump({
    #         "split"      : split,
    #         "annotations": annotations,
    #     }, pkl_file, indent=4)
    return

def statisticize_ball_type_counts():
    ball_type_count = [ 0 ] * 18
    # for video_id in range(1, 800+1):
    for video_id in tqdm(train_formal_list):
        hit_data = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_S2.csv")[["Hitter", "BallType"]].values
        for hitter, ball_type in hit_data:
            if hitter == 'B': ball_type += 9
            ball_type_count[ball_type-1] += 1
    print(ball_type_count, sum(ball_type_count))
    return


if __name__ == "__main__":
    get_video_id_list()
    # plot_first_frames()
    # darken_video(1)
    # crop_video(746)
    # plot_image()
    # plot_background_lines()
    # check_length_match()
    # combine_ball_and_pose_csv_sv1()
    # convert_ground_truth_v1()
    # convert_ground_truth_v2()
    # split_video_into_images()
    # load_images()
    # create_h5py()
    # scale_images()
    # analyze_hits()
    # scale_and_crop_images()
    # crop_images()
    # get_max_frame_count()
    # get_max_hit_count()
    # get_hitter()
    # combine_ball_and_pose_csv_v2()
    # detect_same_frames()
    # create_posec3d_dataset()
    # statisticize_ball_type_counts()
    pass