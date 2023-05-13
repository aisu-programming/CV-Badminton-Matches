import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== #

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

test_formal_list = [
    170, 171, 174, 175, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 188, 189, 190, 191, 192, 193, 195,
    197, 198, 199, 201, 202, 204, 205, 206, 207, 208, 209, 210, 211, 213, 214, 215, 216, 218, 219, 220, 221,
    222, 223, 224, 225, 228, 229, 230, 231, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247,
    248, 249, 250, 251, 252, 253, 255, 256, 257, 258, 259, 260, 262, 263, 264, 265, 266, 267, 268, 269, 270,
    271, 272, 273, 274, 275, 277, 278, 279, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 293, 294, 295,
    296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316,
    317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 331, 332, 333, 334, 335, 336, 337, 338,
    339, 340, 341, 342, 343, 344, 345, 347, 348, 349, 350, 351, 352, 354, 355, 356, 357, 358, 359, 362, 363,
    364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384,
    385, 387, 388, 389, 390, 392, 393, 394, 395, 397, 399
]

test_informal_list = [
    172, 173, 176, 187, 194, 196, 200, 203, 212, 217, 226, 227, 232, 233, 234, 254, 261,
    276, 280, 291, 292, 330, 346, 353, 360, 361, 386, 391, 396, 398
]

# ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== #

# valid_video_id = [
#     335, 624, 776, 381, 489, 720, 558, 796, 655, 374, 736, 99, 724, 755, 622, 449, 301, 194, 203, 400,
#     267, 27, 240, 766, 287, 215, 504, 131, 109, 403, 16, 354, 58, 484, 764, 204, 688, 73, 294, 519,
#     797, 413, 216, 540, 118, 174, 279, 429, 629, 469, 599, 388, 171, 775, 135, 663, 162, 277, 743, 713,
#     404, 252, 153, 54, 499, 530, 163, 686, 465, 258, 89, 565, 293, 285, 161, 324, 772, 18, 619, 691,
#     56, 581, 98, 792, 158, 690, 729, 402, 103, 633, 243, 410, 183, 232, 19, 296, 111, 38, 129, 618, 394,
#     297, 154, 347, 586, 615, 291, 175, 59, 613, 26, 188, 318, 317, 226, 474, 390, 345, 137, 603, 573,
#     372, 719, 510, 497, 543, 165, 542, 134, 707, 590, 97, 705, 361, 191, 187, 206, 220, 342, 343, 694,
#     620, 490, 732, 623, 503, 572, 262, 740, 231, 785, 737, 340, 657, 760, 420, 350, 754, 245, 225,
# ]

# ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== #

def get_video_id_list():
    formal_mp4s = os.listdir("data/test_mp4/informal")
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
    image = cv2.imread("data/background/00012.png")
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

def rename_files():
    for video_id in tqdm(train_formal_list):
        try:
            from_filename = f"data/train/{video_id:05}/prediction_1_hitter.csv" 
            to_filename   = f"data/train/{video_id:05}/{video_id:05}_prediction_1_hitter.csv"
            os.rename(from_filename, to_filename)
            from_filename = f"data/train/{video_id:05}/prediction_1_hitter.png" 
            to_filename   = f"data/train/{video_id:05}/{video_id:05}_prediction_1_hitter.png"
            os.rename(from_filename, to_filename)
        except FileNotFoundError:
            continue
    return

def patch_classification():
    import json
    from data.test_background.classification import img_to_background as test_img_to_background
    # from data.test_background.classification import background_to_img as test_background_to_img
    
    for video_id in test_informal_list: test_img_to_background[video_id] = 13
    test_img_to_background = dict(sorted(test_img_to_background.items(), key=lambda i: i[0]))
    background_id_test_to_train = {
        0: 7,
        1: 5,
        2: 2,
        3: 4,
        4: 3,
        5: 6,
        6: 11,
        7: 1,
        8: 9,
        9: 8,
        10: 0,
        11: 11,
        12: 11,
        13: 12,
    }
    for video_id in range(170, 399+1):
        test_img_to_background[video_id] = background_id_test_to_train[test_img_to_background[video_id]]

    # test_background_to_img_adjusted = {}
    # for background_id in background_id_test_to_train.keys():
    #     test_background_to_img_adjusted[background_id_test_to_train[background_id]] = \
    #         test_background_to_img[background_id]
    # test_background_to_img = test_background_to_img_adjusted
    # test_background_to_img = dict(sorted(test_background_to_img.items(), key=lambda i: i[0]))

    with open(f"outputs/background/avg_without_players/test/type/classification.py", mode='w') as classification_file:
        classification_file.write("img_to_background = ")
        json.dump(test_img_to_background, classification_file, indent=4)
        # classification_file.write("\n\nbackground_to_img = ")
        # json.dump(test_background_to_img, classification_file, indent=4)

def analyze_last_shot():
    last_shot_frame_list, last_shot_diff_list = [], []
    for video_id in tqdm(range(1, 800+1)):
        last_shot_frame = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_S2.csv")[["HitFrame"]].values[-1]
        video_length = len(pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_pose.csv").values)
        last_shot_frame_list.append(last_shot_frame.tolist()[0])
        last_shot_diff_list.append(video_length-last_shot_frame.tolist()[0])
    print("Max last_shot_frame:", max(last_shot_frame_list))
    print("Min last_shot_frame:", min(last_shot_frame_list))
    print("Max last_shot_diff: ", max(last_shot_diff_list))
    print("Min last_shot_diff: ", min(last_shot_diff_list))
    return

def calculate_1_hitter_accuracy():
    valid_video_id_list = [
        335, 624, 776, 381, 489, 720, 558, 796, 655, 374, 736, 99, 724, 755, 622, 449, 301, 194, 203, 400,
        267, 27, 240, 766, 287, 215, 504, 131, 109, 403, 16, 354, 58, 484, 764, 204, 688, 73, 294, 519,
        797, 413, 216, 540, 118, 174, 279, 429, 629, 469, 599, 388, 171, 775, 135, 663, 162, 277, 743, 713,
        404, 252, 153, 54, 499, 530, 163, 686, 465, 258, 89, 565, 293, 285, 161, 324, 772, 18, 619, 691,
        56, 581, 98, 792, 158, 690, 729, 402, 103, 633, 243, 410, 183, 232, 19, 296, 111, 38, 129, 618, 394,
        297, 154, 347, 586, 615, 291, 175, 59, 613, 26, 188, 318, 317, 226, 474, 390, 345, 137, 603, 573,
        372, 719, 510, 497, 543, 165, 542, 134, 707, 590, 97, 705, 361, 191, 187, 206, 220, 342, 343, 694,
        620, 490, 732, 623, 503, 572, 262, 740, 231, 785, 737, 340, 657, 760, 420, 350, 754, 245, 225,
    ]
    total_hit_frame_score, total_hitter_score = [], []
    for video_id in tqdm(valid_video_id_list):
        ground_truth = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_S2.csv")[["HitFrame", "Hitter"]].values
        prediction   = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_prediction_1_hitter.csv")[["HitFrame", "Hitter"]].values
        if len(ground_truth) == len(prediction):
            hit_frame_score, hitter_score = [], []
            for frame_id in range(len(ground_truth)):
                if abs(ground_truth[frame_id, 0]-prediction[frame_id, 0]) <= 2:
                    hit_frame_score.append(0.1)
                    if ground_truth[frame_id, 1] == prediction[frame_id, 1]:
                        hitter_score.append(0.1)
            total_hit_frame_score.append(sum(hit_frame_score)/len(ground_truth))
            total_hitter_score.append(sum(hitter_score)/len(ground_truth))
    print("total_hit_frame_score:", sum(total_hit_frame_score)/len(valid_video_id_list))
    print("total_hitter_score   :", sum(total_hitter_score)/len(valid_video_id_list))
    return

def fill_blank():
    os.makedirs("outputs/final_answer", exist_ok=True)
    dfs = []
    for video_id in tqdm(range(1, 169+1)):
        pred_df_values = pd.read_csv(f"data/valid/{video_id:05}/{video_id:05}_prediction_1_hitter.csv")
        winner         = ['X']*len(pred_df_values)
        winner[-1]     = 'A'
        pred_df_values["VideoName"]         = [f"{video_id:05}.mp4"]*len(pred_df_values)
        pred_df_values["ShotSeq"]           = pred_df_values[["ShotSeq"]] +1
        pred_df_values["RoundHead"]         = [   1 ]*len(pred_df_values)
        pred_df_values["Backhand"]          = [   1 ]*len(pred_df_values)
        pred_df_values["BallHeight"]        = [   1 ]*len(pred_df_values)
        pred_df_values["LandingX"]          = [ 0.0 ]*len(pred_df_values)
        pred_df_values["LandingY"]          = [ 0.0 ]*len(pred_df_values)
        pred_df_values["HitterLocationX"]   = [ 0.0 ]*len(pred_df_values)
        pred_df_values["HitterLocationY"]   = [ 0.0 ]*len(pred_df_values)
        pred_df_values["DefenderLocationX"] = [ 0.0 ]*len(pred_df_values)
        pred_df_values["DefenderLocationY"] = [ 0.0 ]*len(pred_df_values)
        pred_df_values["BallType"]          = [   1 ]*len(pred_df_values)
        pred_df_values["Winner"]            = winner
        dfs.append(pred_df_values)
    # for video_id in tqdm(range(170, 399+1)):
    #     pred_df_values = pd.read_csv(f"data/valid/00002/00002_prediction_1_hitter.csv")
    #     winner         = ['X']*len(pred_df_values)
    #     winner[-1]     = 'A'
    #     pred_df_values["VideoName"]         = [f"{video_id:05}.mp4"]*len(pred_df_values)
    #     pred_df_values["ShotSeq"]           = pred_df_values[["ShotSeq"]] +1
    #     pred_df_values["RoundHead"]         = [   1 ]*len(pred_df_values)
    #     pred_df_values["Backhand"]          = [   1 ]*len(pred_df_values)
    #     pred_df_values["BallHeight"]        = [   1 ]*len(pred_df_values)
    #     pred_df_values["LandingX"]          = [ 0.0 ]*len(pred_df_values)
    #     pred_df_values["LandingY"]          = [ 0.0 ]*len(pred_df_values)
    #     pred_df_values["HitterLocationX"]   = [ 0.0 ]*len(pred_df_values)
    #     pred_df_values["HitterLocationY"]   = [ 0.0 ]*len(pred_df_values)
    #     pred_df_values["DefenderLocationX"] = [ 0.0 ]*len(pred_df_values)
    #     pred_df_values["DefenderLocationY"] = [ 0.0 ]*len(pred_df_values)
    #     pred_df_values["BallType"]          = [   1 ]*len(pred_df_values)
    #     pred_df_values["Winner"]            = winner
    dfs = pd.concat(dfs)
    dfs = dfs.set_index("VideoName")
    dfs.to_csv("outputs/final_answer_valid.csv")
    return

def statisticize_pose_and_location_diff():
    truth_columns = [ "HitFrame", "Hitter", "HitterLocationX", "HitterLocationY",
                                            "DefenderLocationX", "DefenderLocationY" ]
    # pose_columns  = [ "Player A right_ankle X", "Player A right_ankle Y",
    #                   "Player B right_ankle X", "Player B right_ankle Y" ]
    pose_columns  = [ "Player A right_big_toe X", "Player A right_big_toe Y",
                      "Player B right_big_toe X", "Player B right_big_toe Y" ]
    # pose_columns  = [ "Player A right_small_toe X", "Player A right_small_toe Y",
    #                   "Player B right_small_toe X", "Player B right_small_toe Y" ]
    
    total_hit_diff,  total_def_diff  = [], []
    total_hit_score, total_def_score = [], []

    for video_id in tqdm(range(1, 800+1)):
        truth_df_values = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_S2.csv")[truth_columns].values
        # pose_df_values  = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_pose.csv")[pose_columns].values
        pose_df_values  = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_pose_wholebody.csv")[pose_columns].values

        hit_diff, def_diff = [], []
        hit_score, def_score = [], []
        for hit_frame, hitter, hit_x, hit_y, def_x, def_y in truth_df_values:
            pAraX, pAraY, pBraX, pBraY = pose_df_values[hit_frame]
            if hitter=='A':
                pred_hit_x, pred_hit_y = pAraX, pAraY
                pred_def_x, pred_def_y = pBraX, pBraY
            else:
                pred_hit_x, pred_hit_y = pBraX, pBraY
                pred_def_x, pred_def_y = pAraX, pAraY
            if not (np.isnan(pred_hit_x) or np.isnan(pred_hit_y)):
                diff = ((pred_hit_x-hit_x)**2+(pred_hit_y-hit_y)**2)**0.5
                hit_diff.append(diff)
                hit_score.append(diff<10)
            if not (np.isnan(pred_def_x) or np.isnan(pred_def_y)):
                diff = ((pred_def_x-def_x)**2+(pred_def_y-def_y)**2)**0.5
                def_diff.append(diff)
                def_score.append(diff<10)

        # print(video_id, def_diff, def_score)
        total_hit_diff.append(sum(hit_diff)/len(truth_df_values))
        total_def_diff.append(sum(def_diff)/len(truth_df_values))
        total_hit_score.append(sum(hit_score)/len(truth_df_values))
        total_def_score.append(sum(def_score)/len(truth_df_values))

    print("total_hit_diff :", sum(total_hit_diff)/800)
    print("total_def_diff :", sum(total_def_diff)/800)
    print("total_hit_score:", sum(total_hit_score)/800)
    print("total_def_score:", sum(total_def_score)/800)
    return

def plot_pose_and_location():
    truth_columns = [ "HitFrame", "Hitter", "HitterLocationX", "HitterLocationY",
                                            "DefenderLocationX", "DefenderLocationY" ]
    # pose_columns  = [ "Player A right_ankle X", "Player A right_ankle Y",
    #                   "Player B right_ankle X", "Player B right_ankle Y" ]
    pose_columns  = [ "Player A right_big_toe X", "Player A right_big_toe Y",
                      "Player B right_big_toe X", "Player B right_big_toe Y" ]
    # pose_columns  = [ "Player A right_small_toe X", "Player A right_small_toe Y",
    #                   "Player B right_small_toe X", "Player B right_small_toe Y" ]
    
    total_hit_diff,  total_def_diff  = [], []
    total_hit_score, total_def_score = [], []

    for video_id in tqdm(range(1, 800+1)):
        truth_df_values = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_S2.csv")[truth_columns].values
        # pose_df_values  = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_pose.csv")[pose_columns].values
        pose_df_values  = pd.read_csv(f"data/train/{video_id:05}/{video_id:05}_pose_wholebody.csv")[pose_columns].values

        hit_diff, def_diff = [], []
        hit_score, def_score = [], []
        for hit_frame, hitter, hit_x, hit_y, def_x, def_y in truth_df_values:
            pAraX, pAraY, pBraX, pBraY = pose_df_values[hit_frame]
            if hitter=='A':
                pred_hit_x, pred_hit_y = pAraX, pAraY
                pred_def_x, pred_def_y = pBraX, pBraY
            else:
                pred_hit_x, pred_hit_y = pBraX, pBraY
                pred_def_x, pred_def_y = pAraX, pAraY
            if not (np.isnan(pred_hit_x) or np.isnan(pred_hit_y)):
                diff = ((pred_hit_x-hit_x)**2+(pred_hit_y-hit_y)**2)**0.5
                hit_diff.append(diff)
                hit_score.append(diff<10)
            if not (np.isnan(pred_def_x) or np.isnan(pred_def_y)):
                diff = ((pred_def_x-def_x)**2+(pred_def_y-def_y)**2)**0.5
                def_diff.append(diff)
                def_score.append(diff<10)

        # print(video_id, def_diff, def_score)
        total_hit_diff.append(sum(hit_diff)/len(truth_df_values))
        total_def_diff.append(sum(def_diff)/len(truth_df_values))
        total_hit_score.append(sum(hit_score)/len(truth_df_values))
        total_def_score.append(sum(def_score)/len(truth_df_values))

    print("total_hit_diff :", sum(total_hit_diff)/800)
    print("total_def_diff :", sum(total_def_diff)/800)
    print("total_hit_score:", sum(total_hit_score)/800)
    print("total_def_score:", sum(total_def_score)/800)
    return

# ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== #

if __name__ == "__main__":
    # get_video_id_list()
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
    # rename_files()
    # patch_classification()
    # analyze_last_shot()
    # statisticize_pose_and_location_diff()
    # fill_blank()
    pass