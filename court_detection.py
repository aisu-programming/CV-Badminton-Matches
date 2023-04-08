import cv2
import numpy as np
import mmcv
import matplotlib.pyplot as plt


main_colors = [
    (  0,   0,   0),  # Black
    (255, 255, 255),  # White
    (192,  16,  16),  # Red
    ( 80, 192,  80),  # Light Green
    ( 32, 176,  64),  # Dark Green
    ( 32,  48, 192),  # Blue
]

main_colors_white = [
    0,  # Black
    1,  # White
    0,  # Red
    0,  # Light Green
    0,  # Dark Green
    0,  # Blue
]

main_colors_green = [
    0,  # Black
    0,  # White
    0,  # Red
    1,  # Light Green
    1,  # Dark Green
    0,  # Blue
]

def convert_to_main_colors(img):
    img_distances = []
    for mc in main_colors:
        img_tmp = img - mc
        img_tmp = img_tmp ** 2
        img_tmp = np.sum(img_tmp, axis=-1)
        img_distances.append(img_tmp)
    img_distances = np.array(img_distances)
    img_distances = img_distances.transpose(1, 2, 0)
    img_distances = np.argmin(img_distances, axis=-1)

    main_colors_dict = { mc_id: mc for mc_id, mc in enumerate(main_colors) }
    main_colors_img = np.dstack(np.vectorize(main_colors_dict.__getitem__)(img_distances))

    white_dict = { sc_id: sc for sc_id, sc in enumerate(main_colors_white) }
    white_img = np.dstack(np.vectorize(white_dict.__getitem__)(img_distances))
    white_img = white_img.transpose(2, 1, 0)

    green_dict = { sc_id: sc for sc_id, sc in enumerate(main_colors_green) }
    green_img = np.dstack(np.vectorize(green_dict.__getitem__)(img_distances))
    green_img = green_img.transpose(2, 1, 0)

    return main_colors_img, white_img, green_img

def get_court_by_color(green_img, original_img):
    original_img = original_img.copy()
    # green_img = green_img.copy()
    height, _ = green_img.shape[:2]
    court_coordinates = [[320, 680], [960, 680], [800, 400], [480, 400]]  # [x, y]
    for coord_id, (adj_x, adj_y) in zip(range(len(court_coordinates)), [(-1, 1), (1, 1), (1, -1), (-1, -1)]):
        ct = coord_tmp = court_coordinates[coord_id]
        while True:
            xl = x_left  = min((ct[0]-1+5*adj_x), (ct[0]-1+1*adj_x))
            xr = x_right = max((ct[0]-1+5*adj_x), (ct[0]-1+1*adj_x))
            if adj_y == 1:
                yt = y_top    = min((ct[1]-1+1), min((ct[1]-1+20), height-1))
                yb = y_bottom = max((ct[1]-1+1), min((ct[1]-1+20), height-1))
            else:
                yt = y_top    = min((ct[1]-1-1), min((ct[1]-1-5), height-1))
                yb = y_bottom = max((ct[1]-1-1), min((ct[1]-1-5), height-1))
            if coord_id in [0, 1]:
                if green_img[ct[1], xl:xr].any():
                    ct[0] += 1*adj_x
                elif ct[1] < (height-1) and green_img[yt:yb, ct[0]].any():
                    ct[1] += 1*adj_y
                else: break
            else:
                if ct[1] < (height-1) and green_img[yt:yb, ct[0]].any():
                    ct[1] += 1*adj_y
                elif green_img[ct[1], xl:xr].any():
                    ct[0] += 1*adj_x
                else: break
            cv2.circle(original_img, ct, 2, (255, 0, 0), 2)
        court_coordinates[coord_id] = ct
    # cv2.circle(original_img, [320, 680], 1, (255, 0, 0), 1)
    # cv2.polylines(original_img, np.array([initial_court_polygon]), True, (0,0,0), 5)
    return original_img

def get_court_line_by_line_fitting(white_img, original_img):
    line_mask = np.zeros_like(white_img)
    line_mask[320:960, 680] = 1
    print(line_mask)

for video_id in range(1, 800+1):

    # img0 = mmcv.VideoReader(f"data/train/{video_id:05}/{video_id:05}.mp4")[0]
    # img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    # img1 = img0[:,:,1]
    # img2 = img1 > (255-32)
    # img3 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
    # img4 = img3 > (255-32)
    # img5 = img0[:,:,1]
    # minus = img0[:,:,0]*(img0[:,:,0]>=img0[:,:,2]) + img0[:,:,2]*(img0[:,:,0]<img0[:,:,2])
    # img5 = img0[:,:,1] - minus*(img0[:,:,1]>=minus) - img0[:,:,1]*(img0[:,:,1]<minus)
    # img6 = img5 > 20
    # img7 = (img6 + img4) > 0
    # plt.subplot(331).imshow(img0)
    # plt.subplot(332).imshow(img1)
    # plt.subplot(333).imshow(img2)
    # plt.subplot(335).imshow(img3)
    # plt.subplot(336).imshow(img4)
    # plt.subplot(338).imshow(img5)

    original_img = mmcv.VideoReader(f"data/train/{video_id:05}/{video_id:05}.mp4")[0]
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    main_colors_img, white_img, green_img = convert_to_main_colors(original_img)
    # court_polygon_img = get_court_by_color(green_img, original_img)

    get_court_line_by_line_fitting(white_img, original_img)

    plt.suptitle(f"{video_id:05}.mp4", fontsize=16)

    plt.subplot(231)
    plt.imshow(original_img)
    plt.title("original_img")
    plt.axis('off')

    plt.subplot(234)
    plt.title("main_colors_img")
    plt.imshow(main_colors_img)
    plt.axis('off')

    plt.subplot(232)
    plt.title("white_img")
    plt.imshow(white_img)
    plt.axis('off')

    plt.subplot(233)
    plt.title("green_img")
    plt.imshow(green_img)
    plt.axis('off')

    plt.tight_layout()
    plt.show()