import cv2
import numpy as np
import mmcv
import matplotlib.pyplot as plt

from color import all_colors, all_colors_green, all_colors_white, \
                  court_colors, court_colors_green, court_colors_white


# Archieved
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

def classify_colors(img, court=False):
    
    main_colors       = court_colors       if court else all_colors
    main_colors_white = court_colors_white if court else all_colors_white
    main_colors_green = court_colors_green if court else all_colors_green

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

def get_court_by_diff(normalized_court_img, court_mask):

    normalized_court_img = np.float32(normalized_court_img)
    court_mask           = np.float32(court_mask)

    kernel = np.ones((11, 11))
    avg_court_img  = cv2.filter2D(normalized_court_img, 3, kernel=kernel)
    avg_court_mask = np.expand_dims(cv2.filter2D(court_mask, 1, kernel=kernel), axis=-1)
    avg_court_img  = avg_court_img / avg_court_mask
    avg_court_img  = np.nan_to_num(avg_court_img)
    avg_court_img  = np.uint8(avg_court_img)
    classified_court_mask = np.zeros_like(court_mask)
    classified_court_mask[ np.where(court_mask) ] += 1
    classified_court_mask[ np.where(np.greater(np.sum((normalized_court_img-avg_court_img), axis=-1), 50)) ] += 1
    white_mask = np.zeros_like(court_mask, dtype=np.uint8)
    white_mask[classified_court_mask==2] += 1
    white_mask = np.float32(white_mask)
    kernel = np.array([[0.2, 0.2, 0.2],
                       [0.2, 1.0, 0.2],
                       [0.2, 0.2, 0.2]], dtype=np.float32)
    white_mask = cv2.filter2D(white_mask, -1, kernel)
    white_mask[white_mask>=0.9] = 1
    white_mask[white_mask<0.9]  = 0
    white_mask = np.uint8(white_mask)
    return white_mask

def get_both_ends(y_top, white_mask):
    y_thick, xl, xr = 12, 480, 800
    while True:
        now_covered_area = np.logical_or.reduce([
            white_mask[y_top+yt:y_top+yt+1, xl:xr]
            for yt in range(int(y_thick/-2), int(y_thick/2))
        ]).sum()
        xl_expand_covered_area = np.logical_or.reduce([
            white_mask[y_top+yt:y_top+yt+1, xl-2:xr]
            for yt in range(int(y_thick/-2), int(y_thick/2))
        ]).sum()
        xr_expand_covered_area = np.logical_or.reduce([
            white_mask[y_top+yt:y_top+yt+1, xl:xr+2]
            for yt in range(int(y_thick/-2), int(y_thick/2))
        ]).sum()
        if xl_expand_covered_area > now_covered_area: xl -= 1
        elif xr_expand_covered_area > now_covered_area: xr += 1
        else: break
    gk = gaussian_kernel = (17, 7)
    ye = y_extend = 6
    xe = x_extend = 12
    coord_left, coord_right = [ xl, y_top ], [ xr, y_top ]
    blur = cv2.GaussianBlur(white_mask[y_top-ye:y_top+ye, xl-xe:xl+xe], gk, 0)
    coord_left_blur  = [ xl-xe+np.argmax(blur)%(xe*2), y_top-ye+(np.argmax(blur)//(xe*2)) ]
    blur = cv2.GaussianBlur(white_mask[y_top-ye:y_top+ye, xr-xe:xr+xe], gk, 0)
    coord_right_blur = [ xr-xe+np.argmax(blur)%(xe*2), y_top-ye+(np.argmax(blur)//(xe*2)) ]
    return coord_left, coord_right, coord_left_blur, coord_right_blur

def get_court_line_by_line_fitting(white_mask):

    white_mask = np.array(white_mask, dtype=np.float32)
    y_top, y_thick, y_range = 680, 5, 400
    xl, xr = 360, 920

    # Get the cover area and squeeze to an array with width 1 by or operation
    covers = np.array([
        np.logical_or.reduce([
            white_mask[y_top-yr+yt:y_top-yr+yt+1, int(xl+yr/3):int(xr-yr/3)]
            for yt in range(y_thick)
        ]).sum()
        for yr in range(y_range)
    ])

    best_y_tops_amount = 4
    best_y_tops = []
    for _ in range(best_y_tops_amount):
        best_y_tops.append(y_top-np.argmax(covers))
        covers[max(0, np.argmax(covers)-15):min(np.argmax(covers)+15, y_range)] = 0  # Remove the already-selected y_top
    best_y_tops.sort()

    line_masks = []
    for byt_id in range(best_y_tops_amount):
        lm = line_mask = np.zeros_like(white_mask, dtype=np.float32)
        lm[best_y_tops[byt_id]:best_y_tops[byt_id]+y_thick, int(xl+(y_top-best_y_tops[byt_id])/4):int(xr-(y_top-best_y_tops[byt_id])/4)] += 1.0
        line_masks.append(lm)

    # Get both ends of the opponent's front service line
    ofsll, ofslr, ofsllb, ofslrb = get_both_ends(best_y_tops[0], white_mask)  # ofsll = opponent_front_service_line_left
                                                                              # ofslr = opponent_front_service_line_right
    # Get both ends of the front service line
    fsll, fslr, fsllb, fslrb = get_both_ends(best_y_tops[1], white_mask)      # fsll = front_service_line_left
                                                                              # fslr = front_service_line_right
    # Get both ends of the doubles back service line
    dbsll, dbslr, dbsllb, dbslrb = get_both_ends(best_y_tops[2], white_mask)  # dbsll = doubles_back_service_line_left
                                                                              # dbslr = doubles_back_service_line_right
    # Get the left end of the singles back service line
    sbsll, sbslr, sbsllb, sbslrb = get_both_ends(best_y_tops[3], white_mask)  # sbsll = singles_back_service_line_left
                                                                              # sbslr = singles_back_service_line_right

    return [ ofsll, ofslr, ofsllb, ofslrb, fsll, fslr, fsllb, fslrb, dbsll, dbslr, dbsllb, dbslrb, sbsll, sbslr, sbsllb, sbslrb ]


from misc import train_formal_list

problem_list = [ 197, 207, 278, 446, 453, 688 ]

counter, amount = 0, 10
for video_id in train_formal_list:

    if video_id in [ 2, 64, 153, 156, 170, 212, 290, 293, 378, 431, 527, 678, 697, 711, 712, 722 ]:
        original_img = mmcv.VideoReader(f"data/train/{video_id:05}/{video_id:05}.mp4")[-1]
    elif video_id in [ 322, 663, 727 ]:
        original_img = mmcv.VideoReader(f"data/train/{video_id:05}/{video_id:05}.mp4")[50]
    elif video_id in [ 717, 792 ]:
        original_img = mmcv.VideoReader(f"data/train/{video_id:05}/{video_id:05}.mp4")[150]
    else:
        original_img = mmcv.VideoReader(f"data/train/{video_id:05}/{video_id:05}.mp4")[0]
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    all_colors_img, white_mask, green_mask = classify_colors(original_img)
    # court_polygon_img = get_court_by_color(green_mask, original_img)

    court_mask = np.float32(white_mask+green_mask)
    # court_mask_adjusted = np.expand_dims(cv2.filter2D(court_mask, -1, kernel=np.ones((5, 5))), axis=-1)
    # court_mask_adjusted[court_mask_adjusted<20] = 0
    # court_mask_adjusted[court_mask_adjusted>=20] = 1
    # court_img = original_img * court_mask_adjusted
    court_img = original_img * court_mask

    normalized_court_img = cv2.normalize(court_img, None, 0, 255, cv2.NORM_MINMAX)
    # green_mask_adjusted, white_mask_adjusted = get_court_by_diff(normalized_court_img, court_mask_adjusted)
    white_mask_adjusted = get_court_by_diff(normalized_court_img, court_mask)

    print(video_id)

    ofsll, ofslr, ofsllb, ofslrb, \
        fsll, fslr, fsllb, fslrb, \
        dbsll, dbslr, dbsllb, dbslrb, \
        sbsll, sbslr, sbsllb, sbslrb = get_court_line_by_line_fitting(white_mask_adjusted)

    if counter % amount == 0:
        plt.figure(figsize=(36, 2*amount))
        # plt.suptitle(f"{video_id:05}.mp4", fontsize=16)

    plt.subplot(amount, 13, 13*(counter%amount)+1)
    plt.title(f"{video_id:05}.mp4")
    plt.imshow(original_img)
    plt.axis("off")

    plt.subplot(amount, 13, 13*(counter%amount)+2)
    plt.imshow(all_colors_img)
    plt.axis("off")

    # plt.subplot(amount, 13, 13*(counter%amount)+3)
    # plt.imshow(white_mask)
    # plt.axis("off")

    # plt.subplot(amount, 13, 13*(counter%amount)+4)
    # plt.imshow(green_mask)
    # plt.axis("off")

    # plt.subplot(amount, 13, 13*(counter%amount)+5)
    # plt.imshow(court_mask)
    # plt.axis("off")

    # plt.subplot(amount, 13, 13*(counter%amount)+6)
    # plt.imshow(court_mask_adjusted)
    # plt.axis("off")

    # plt.subplot(amount, 13, 13*(counter%amount)+7)
    # plt.imshow(court_img)
    # plt.axis("off")

    # plt.subplot(amount, 13, 13*(counter%amount)+8)
    # plt.imshow(normalized_court_img)
    # plt.axis("off")

    plt.subplot(amount, 13, 13*(counter%amount)+3)
    plt.imshow(white_mask_adjusted)
    plt.axis("off")

    # plt.subplot(amount, 13, 13*(counter%amount)+10)
    # plt.imshow(green_mask_adjusted)
    # plt.axis("off")

    plt.subplot(amount, 13, 13*(counter%amount)+4)
    plt.imshow(white_mask_adjusted)
    for coordl, coordr in [ (ofsll, ofslr), (fsll, fslr), (dbsll, dbslr), (sbsll, sbslr) ]:
        plt.plot((coordl[0], coordr[0]), (coordl[1], coordr[1]), marker='o', linewidth=1, markersize=2)
    plt.axis([sbsll[0]-30, sbslr[0]+30, sbsll[1]+50, ofslr[1]-50])
    plt.axis("off")
    
    plt.subplot(amount, 13, 13*(counter%amount)+5)
    plt.imshow(white_mask_adjusted)
    for coordl, coordr in [ (ofsllb, ofslrb), (fsllb, fslrb), (dbsllb, dbslrb), (sbsllb, sbslrb) ]:
        plt.plot((coordl[0], coordr[0]), (coordl[1], coordr[1]), marker='o', linewidth=1, markersize=2)
    plt.axis([sbsllb[0]-30, sbslrb[0]+30, sbsllb[1]+50, ofslrb[1]-50])
    plt.axis("off")
    
    for ctr, coord in zip(range(8), [ofsllb, ofslrb, fsllb, fslrb, dbsllb, dbslrb, sbsllb, sbslrb]):
        plt.subplot(amount, 13, 13*(counter%amount)+6+ctr)
        plt.imshow(white_mask_adjusted)
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