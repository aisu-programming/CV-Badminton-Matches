import cv2
import numpy as np
import mmcv
import matplotlib.pyplot as plt

from color import main_colors, main_colors_green, main_colors_white


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

def get_both_ends(y_top, white_img):
    y_thick, xl, xr = 15, 480, 800
    while True:
        now_covered_area = np.logical_or.reduce([
            white_img[y_top+yt:y_top+yt+1, xl:xr]
            for yt in range(int(y_thick/-2), int(y_thick/2))
        ]).sum()
        xl_expand_covered_area = np.logical_or.reduce([
            white_img[y_top+yt:y_top+yt+1, xl-5:xr]
            for yt in range(int(y_thick/-2), int(y_thick/2))
        ]).sum()
        xr_expand_covered_area = np.logical_or.reduce([
            white_img[y_top+yt:y_top+yt+1, xl:xr+5]
            for yt in range(int(y_thick/-2), int(y_thick/2))
        ]).sum()
        if xl_expand_covered_area > now_covered_area: xl -= 1
        elif xr_expand_covered_area > now_covered_area: xr += 1
        else: break
    gk = gaussian_kernel = (17, 7)
    ye = y_extend = 6
    xe = x_extend = 12
    coord_left, coord_right = [ xl, y_top ], [ xr, y_top ]
    blur = cv2.GaussianBlur(white_img[y_top-ye:y_top+ye, xl-xe:xl+xe], gk, 0)
    coord_left_blur  = [ xl-xe+np.argmax(blur)%(xe*2), y_top-ye+(np.argmax(blur)//(xe*2)) ]
    blur = cv2.GaussianBlur(white_img[y_top-ye:y_top+ye, xr-xe:xr+xe], gk, 0)
    coord_right_blur = [ xr-xe+np.argmax(blur)%(xe*2), y_top-ye+(np.argmax(blur)//(xe*2)) ]
    return coord_left, coord_right, coord_left_blur, coord_right_blur

def get_court_line_by_line_fitting(white_img, original_img):
    original_img = np.array(original_img, dtype=np.float32) / 255

    white_img = np.array(white_img, dtype=np.float32)
    y_top, y_thick, y_range = 700, 5, 400
    xl, xr = 360, 920

    # Get the cover area and squeeze to an array with width 1 by or operation
    covers = np.array([
        np.logical_or.reduce([
            white_img[y_top-yr+yt:y_top-yr+yt+1, int(xl+yr/3):int(xr-yr/3)]
            for yt in range(y_thick)
        ]).sum()
        for yr in range(y_range)
    ])

    best_y_tops_amount = 4
    best_y_tops = []
    for _ in range(best_y_tops_amount):
        best_y_tops.append(y_top-np.argmax(covers))
        covers[max(0, np.argmax(covers)-10):min(np.argmax(covers)+10, y_range)] = 0  # Remove the already-selected y_top
    best_y_tops.sort()

    line_masks = []
    for byt_id in range(best_y_tops_amount):
        lm = line_mask = np.zeros_like(white_img, dtype=np.float32)
        lm[best_y_tops[byt_id]:best_y_tops[byt_id]+y_thick, int(xl+(y_top-best_y_tops[byt_id])/4):int(xr-(y_top-best_y_tops[byt_id])/4)] += 1.0
        line_masks.append(lm)

    # Get both ends of the opponent's front service line
    ofsll, ofslr, ofsllb, ofslrb = get_both_ends(best_y_tops[0], white_img)  # ofsll = opponent_front_service_line_left
                                                                             # ofslr = opponent_front_service_line_right
    # Get both ends of the front service line
    fsll, fslr, fsllb, fslrb = get_both_ends(best_y_tops[1], white_img)      # fsll = front_service_line_left
                                                                             # fslr = front_service_line_right
    # Get both ends of the doubles back service line
    dbsll, dbslr, dbsllb, dbslrb = get_both_ends(best_y_tops[2], white_img)  # dbsll = doubles_back_service_line_left
                                                                             # dbslr = doubles_back_service_line_right
    # Get the left end of the singles back service line
    sbsll, sbslr, sbsllb, sbslrb = get_both_ends(best_y_tops[3], white_img)  # sbsll = singles_back_service_line_left
                                                                             # sbslr = singles_back_service_line_right

    # badminton_court_img = cv2.imread("badminton_court_2.jpg")
    # badminton_court_keypoint = np.array([[13.5, 569], [471.5, 569], [13.5, 828], [471.5, 828]])
    # h, status = cv2.findHomography(badminton_court_keypoint, np.array([fsll, fslr, dbsll, dbslr]))
    # badminton_court_perspective = cv2.warpPerspective(badminton_court_img, h, (original_img.shape[1], original_img.shape[0]))

    # # Display images
    # badminton_court_perspective = np.array(badminton_court_perspective / 255, dtype=np.float32)
    # badminton_court_perspective = cv2.addWeighted(original_img, 0.8, badminton_court_perspective, 1, 0)
    # cv2.imshow("", badminton_court_perspective)
    # cv2.waitKey()

    return [ ofsll, ofslr, ofsllb, ofslrb, fsll, fslr, fsllb, fslrb, dbsll, dbslr, dbsllb, dbslrb, sbsll, sbslr, sbsllb, sbslrb ]


from misc import formal_list

problem_list = [ 197, 207, 278, 446, 453, 688 ]

counter, amount = 0, 10
for video_id in formal_list:

    if video_id in [ 2, 64, 153, 156, 170, 212, 290, 293, 378, 431, 527, 678, 697, 711, 712, 722 ]:
        original_img = mmcv.VideoReader(f"data/train/{video_id:05}/{video_id:05}.mp4")[-1]
    elif video_id in [ 322, 663, 727 ]:
        original_img = mmcv.VideoReader(f"data/train/{video_id:05}/{video_id:05}.mp4")[50]
    elif video_id in [ 717, 792 ]:
        original_img = mmcv.VideoReader(f"data/train/{video_id:05}/{video_id:05}.mp4")[150]
    else:
        original_img = mmcv.VideoReader(f"data/train/{video_id:05}/{video_id:05}.mp4")[0]
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    main_colors_img, white_img, green_img = convert_to_main_colors(original_img)
    # court_polygon_img = get_court_by_color(green_img, original_img)
    print(video_id)
    ofsll, ofslr, ofsllb, ofslrb, \
        fsll, fslr, fsllb, fslrb, \
        dbsll, dbslr, dbsllb, dbslrb, \
        sbsll, sbslr, sbsllb, sbslrb = get_court_line_by_line_fitting(white_img, original_img)

    if counter % amount == 0:
        plt.figure(figsize=(36, 2*amount))
        # plt.suptitle(f"{video_id:05}.mp4", fontsize=16)

    plt.subplot(amount, 13, 13*(counter%amount)+1)
    plt.title(f"{video_id:05}.mp4")
    plt.imshow(original_img)
    plt.axis("off")

    plt.subplot(amount, 13, 13*(counter%amount)+2)
    plt.imshow(main_colors_img)
    plt.axis("off")

    plt.subplot(amount, 13, 13*(counter%amount)+3)
    plt.imshow(white_img)
    plt.axis("off")

    plt.subplot(amount, 13, 13*(counter%amount)+4)
    plt.imshow(white_img)
    for coordl, coordr in [ (ofsll, ofslr), (fsll, fslr), (dbsll, dbslr), (sbsll, sbslr) ]:
        plt.plot((coordl[0], coordr[0]), (coordl[1], coordr[1]), marker='o', linewidth=1, markersize=2)
    plt.axis([sbsll[0]-30, sbslr[0]+30, sbsll[1]+50, ofslr[1]-50])
    plt.axis("off")
    
    plt.subplot(amount, 13, 13*(counter%amount)+5)
    plt.imshow(white_img)
    for coordl, coordr in [ (ofsllb, ofslrb), (fsllb, fslrb), (dbsllb, dbslrb), (sbsllb, sbslrb) ]:
        plt.plot((coordl[0], coordr[0]), (coordl[1], coordr[1]), marker='o', linewidth=1, markersize=2)
    plt.axis([sbsllb[0]-30, sbslrb[0]+30, sbsllb[1]+50, ofslrb[1]-50])
    plt.axis("off")
    
    for ctr, coord in zip(range(8), [ofsllb, ofslrb, fsllb, fslrb, dbsllb, dbslrb, sbsllb, sbslrb]):
        plt.subplot(amount, 13, 13*(counter%amount)+6+ctr)
        plt.imshow(white_img)
        plt.plot((ofsllb[0], ofslrb[0]), (ofsllb[1], ofslrb[1]), marker='o', linewidth=3, markersize=8)
        plt.plot((fsllb[0], fslrb[0]), (fsllb[1], fslrb[1]), marker='o', linewidth=3, markersize=8)
        plt.plot((dbsllb[0], dbslrb[0]), (dbsllb[1], dbslrb[1]), marker='o', linewidth=3, markersize=8)
        plt.plot((sbsllb[0], sbslrb[0]), (sbsllb[1], sbslrb[1]), marker='o', linewidth=3, markersize=8)
        plt.axis([coord[0]-16, coord[0]+16, coord[1]+9, coord[1]-9])
        plt.axis("off")

    if (counter%amount==amount-1):  # or (video_id==formal_list[-1]):
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"court_output/{int(counter/amount):03}", dpi=500)
        plt.close()

    counter += 1
    
plt.tight_layout()
# plt.show()
plt.savefig(f"court_output/{int(counter/amount):03}", dpi=500)
plt.close()