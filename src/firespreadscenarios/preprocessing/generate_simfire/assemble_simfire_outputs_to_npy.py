# Takes the outputs of generate_simfire_dataset.py and assembles them into numpy arrays.
import glob
import sys

import cv2
import numpy as np
import tqdm

# path to 'data' folder
root_dir = sys.argv[1]


def load_files_from_dir(d):
    npy_files = {f: np.load(f"{d}/{f}") for f in npy_filenames}
    png_files = {f: cv2.imread(f"{d}/{f}", 0) for f in png_filenames}
    return npy_files, png_files


def collate_x_y(npy_dict, png_dict):
    x_list = list(npy_dict.values())
    x_list.append(png_dict.pop("fire_pre_0.png"))
    y_list = list(png_dict.values())

    x = np.stack(x_list)
    y = np.stack(y_list)
    return x, y


png_filenames = [
    "fire_pre_0.png",
    "fire_post_0.png",
    "fire_post_1.png",
    "fire_post_2.png",
    "fire_post_3.png",
    "fire_post_4.png",
    "fire_post_5.png",
    "fire_post_6.png",
    "fire_post_7.png",
]
npy_filenames = [
    "M_x.npy",
    "delta.npy",
    "elevation.npy",
    "sigma.npy",
    "w_0.npy",
    "wind_speed.npy",
]

data_dirs = glob.glob(root_dir + "*")
print("len data dirs: ", len(data_dirs))

xs = []
ys = []
skip_count = 0
for i in tqdm.tqdm(range(len(data_dirs))):
    npy_dict, png_dict = load_files_from_dir(data_dirs[i])
    if 128 in png_dict["fire_pre_0.png"]:
        x, y = collate_x_y(npy_dict, png_dict)
        xs.append(x)
        ys.append(y)
    else:
        skip_count += 1
print(f"{skip_count=}")

X = np.stack(xs)
Y = np.stack(ys)

# Normalize each channel to [0,1] using statistics from the first 5000 images, which will constitute the training set.
x_max = X[:5000].max((0, 2, 3))[None, :, None, None]
x_min = X[:5000].min((0, 2, 3))[None, :, None, None]
X_norm = (X - x_min) / (x_max - x_min)
Y_norm = (Y > 0).astype(int)

np.save(root_dir + "X.npy", X_norm)
np.save(root_dir + "Y.npy", Y_norm)
