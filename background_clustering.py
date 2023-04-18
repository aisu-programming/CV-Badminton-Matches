import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from misc import train_formal_list


bg_t_imgs = background_type_images = []
bg_cls    = background_classification = {}

for video_id in tqdm(train_formal_list, desc="Clustering backgrounds"):

    img = cv2.imread(f"output/background/avg_without_players/{video_id:05}.png")

    if len(bg_t_imgs) == 0:
        bg_t_imgs.append(img)
        bg_cls[video_id] = len(bg_t_imgs)-1
        continue

    distance = np.sum((np.array(bg_t_imgs)-img)**2, axis=(1, 2, 3))
    is_alike = distance < 140000000

    if is_alike.any():
        bg_cls[video_id] = int(np.argmin(distance))
        bg_amount = np.sum(np.array(list(bg_cls.values()))==np.argmin(distance))
        bgtimg = np.copy(bg_t_imgs[np.argmin(distance)])
        bafb = bg_amount_for_bgtimg = np.reshape((bg_amount-1) * (np.sum(bgtimg, axis=-1) > 0), (720, 1280, 1))
        bafn = bg_amount_for_newimg = np.reshape(1 * (np.sum(img, axis=-1) > 0), (720, 1280, 1))
        bga  = bg_amount = np.maximum(bafb+bafn, 1)
        bg_t_imgs[np.argmin(distance)] = np.uint8(bgtimg*(bafb/bga) + img*(bafn/bga))
    else:
        bg_t_imgs.append(img)
        bg_cls[video_id] = len(bg_t_imgs)-1

for filename in os.listdir("output/background/avg_without_players/type"):
    os.remove(f"output/background/avg_without_players/type/{filename}")
os.removedirs("output/background/avg_without_players/type")
os.makedirs("output/background/avg_without_players/type", exist_ok=True)

for bgtimg_id, bgtimg in enumerate(bg_t_imgs):
    cv2.imwrite(f"output/background/avg_without_players/type/{bgtimg_id:05}.png", bgtimg)

with open("output/background/avg_without_players/type/classification.py", mode='w') as classification_file:
    classification_file.write("img_to_background = ")
    json.dump(bg_cls, classification_file, indent=4)
    classification_file.write("\n\nbackground_to_img = ")
    json.dump({
        btimg_id: list(dict(filter(lambda item: item[1]==btimg_id, bg_cls.items())).keys())
        for btimg_id in range(len(bg_t_imgs))
    }, classification_file, indent=4)