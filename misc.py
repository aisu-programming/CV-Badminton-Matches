import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

train_formal_list = [
    1, 2, 3, 4, 5, 8, 9, 10, 11, 14, 17, 22, 23, 24, 25, 26, 27, 28, 29, 30, 
    31, 32, 33, 34, 35, 38, 39, 40, 41, 42, 45, 46, 47, 48, 49, 50, 52, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 68, 69, 70, 71, 72, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 93, 94, 95, 96, 97, 98, 99, 101, 102, 103, 104, 105, 107, 108, 109, 110, 111, 112, 113, 114, 115, 117, 118, 119, 120, 121, 122, 123, 125, 126, 127, 128, 129, 130, 131, 132, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 146, 147, 148, 149, 150, 151, 153, 154, 155, 156, 157, 158, 159, 160, 161, 163, 165, 166, 167, 168, 169, 170, 172, 173, 174, 176, 177, 178, 179, 180, 181, 182, 184, 185, 189, 190, 191, 192, 194, 195, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 211, 212, 213, 214, 215, 216, 217, 218, 221, 222, 223, 224, 225, 226, 227, 228, 229, 231, 232, 234, 235, 236, 238, 239, 240, 241, 242, 243,
    244, 246, 247, 249, 253, 254, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 292, 293, 295, 296, 297, 298, 301, 302, 303, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 316, 317, 319, 320, 321, 322, 323, 325, 327, 328, 330, 331, 332, 333, 335, 336, 337, 340, 341, 343, 344, 345, 346, 347, 348, 350, 351, 352, 353, 355, 356, 357, 358, 361, 362, 363, 364, 365, 366, 367, 368, 369, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 390, 391, 392, 393, 394, 395, 396, 398, 399, 400, 401, 402, 403, 405, 406, 407, 408, 409, 410, 411, 412, 413, 415, 416, 417, 418, 419, 421, 422, 423, 424, 425, 426, 427, 428, 429, 431, 432, 435, 436, 437, 438, 440, 442, 443, 445, 446, 447, 449, 452, 453, 454, 456, 457, 459, 461, 462, 464, 467, 468, 470, 473,
    474, 475, 476, 477, 478, 479, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 494, 495, 496, 497, 498, 499, 500, 502, 503, 505, 507, 510, 511, 512, 513, 515, 516, 517, 519, 521, 522, 524, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 547, 548, 549, 550, 551, 552, 553, 556, 557, 558, 559, 561, 562, 563, 564, 565, 566, 567, 570, 571, 572, 573, 574, 575, 577, 580, 581, 582, 583, 584, 585, 587, 588, 589, 593, 594, 595, 598, 599, 600, 601, 602, 603, 604, 605, 606, 608, 609, 610, 611, 613, 614, 615, 617, 618, 619, 620, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 634, 635, 636, 637, 638, 639, 641, 642, 644, 645, 646, 647, 648, 649, 650, 651, 653, 654, 655, 657, 658, 660, 662, 663, 664, 666, 667, 668, 669, 670, 672, 673, 674, 675, 676, 678, 679, 680, 681, 682, 683, 684, 685, 686, 688, 689, 690, 691, 692, 693, 694, 695, 697,
    698, 699, 700, 701, 703, 704, 706, 707, 709, 711, 712, 713, 715, 716, 717, 718, 720, 721, 722, 723, 724, 725, 726, 727, 728, 731, 732, 733, 734, 735, 736, 737, 739, 740, 741, 744, 745, 746, 747, 748, 749, 750, 751, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 791, 792, 794, 795, 796, 797, 798, 799, 800
]

val_foraml_list = [
    1, 3, 4, 5, 6, 7, 9, 11, 12, 13, 14, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 
    50, 52, 53, 54, 55, 57, 58, 59, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 76, 77, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 90, 91, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 115, 116, 117, 118, 119, 123, 124, 125, 127, 128, 129, 130, 131, 132, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 155, 157, 158, 159, 161, 162, 163, 164, 166, 167, 169
]

def get_formal_list():
    formal_mp4s = os.listdir("data/val_mp4/formal")
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

def patch_ball_csv():
    for video_id in tqdm(train_formal_list):
        video_ori_frame_count  = cv2.VideoCapture(f"data/train/{video_id:05}/{video_id:05}.mp4").get(cv2.CAP_PROP_FRAME_COUNT)
        video_ball_frame_count = len(pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_ball_33_adj.csv").values)
        if video_ori_frame_count != video_ball_frame_count:
            with open(f"data/train/{video_id:05}/{video_id:05}_ball_33_adj.csv", mode='a') as csv_file:
                for i in range(int(video_ori_frame_count-video_ball_frame_count)):
                    csv_file.write(str(video_ball_frame_count+i))
                    csv_file.write(',0'*2 + ',0.0' + ',0'*2 + ','*6 + '\n')
        video_ball_frame_count = len(pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_ball_33_adj.csv").values)
        assert video_ori_frame_count == video_ball_frame_count
    return

def combine_ball_and_pose_csv():
    for video_id in tqdm(train_formal_list):
        pose_csv = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_pose.csv")
        ball_csv = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_ball_33_adj.csv")
        # pose_csv = pose_csv.drop([ "Frame", "Player A confidence", "Player B confidence" ], axis=1)
        pose_csv = pose_csv.drop([ "Frame" ], axis=1)
        pose_csv_columns = pose_csv.columns
        for column in pose_csv_columns:
            if   " X" in column: pose_csv[[column]] = pose_csv[[column]] / 1280
            elif " Y" in column: pose_csv[[column]] = pose_csv[[column]] / 720
        ball_csv = ball_csv[[ "Visibility", "X", "Y" ]]
        ball_csv[["X"]] = ball_csv[["X"]] / 1280
        ball_csv[["Y"]] = ball_csv[["Y"]] / 720
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

def crop_images():
    os.makedirs("data/train_background/cropped", exist_ok=True)
    for bg_filename in os.listdir("data/train_background"):
        if ".png" in bg_filename:
            img = cv2.imread(f"data/train_background/{bg_filename}")
            img = cv2.copyMakeBorder(img, 140, 140, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
            img = img[:, 140:-140]
            img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(f"data/train_background/cropped/{bg_filename}", img)
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


if __name__ == "__main__":
    # get_formal_list()
    # plot_first_frames()
    # darken_video(1)
    # crop_video(746)
    # plot_image()
    # plot_background_lines()
    # check_length_match()
    # patch_ball_csv()
    # combine_ball_and_pose_csv()
    # convert_ground_truth_v1()
    # convert_ground_truth_v2()
    # split_video_into_images()
    # load_images()
    # create_h5py()
    # scale_images()
    # crop_images()
    # analyze_hits()
    scale_and_crop_images()
    pass