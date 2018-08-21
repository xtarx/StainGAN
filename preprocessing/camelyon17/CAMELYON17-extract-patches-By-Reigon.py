import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os.path as osp
import openslide
from pathlib import Path
from skimage.filters import threshold_otsu
from openslide.deepzoom import DeepZoomGenerator
import os, errno, sys
from glob import glob
from datetime import datetime
from sklearn.feature_extraction import image
import matplotlib.image as matlabimg

# BASE_TRUTH_DIR = Path('/home/tarek/Downloads/camelyon17/center_4/')
# SLIDE_NAME = 'patient_090_node_1'

BASE_TRUTH_DIR = Path('/home/tarek/Downloads/Train_Tumor/')
SLIDE_NAME = 'Tumor_083'

SLIDE_NAME_EXT = str(SLIDE_NAME) + '.tif'
exp_folder_name = '/' + str(SLIDE_NAME) + '/'
# x = 256 * 80
# y = 256 * 200

x = 256 * 180
y = 256 * 30
zoom_level=3

def load_data_region(slide_name):
    slide_path = str(BASE_TRUTH_DIR / slide_name)
    return make_patches_from_region(slide_path)


def make_patches_from_region(slide_path):
    with openslide.open_slide(slide_path) as slide:
        # thumbnail = slide.read_region((x, y), zoom_level, (4500, 4500))
        # print(slide.get_best_level_for_downsample(1000))
        # sys.exit()
        thumbnail = slide.read_region((x, y), zoom_level, (1000, 1000))

        # thumbnail = slide.get_thumbnail((slide.dimensions[0] / 256, slide.dimensions[1] / 256))

    patches_dir = str(BASE_TRUTH_DIR) + str(exp_folder_name) + '/'
    print('patches_dir', patches_dir)
    assure_path_exists(patches_dir)

    plt.imshow(thumbnail);
    plt.show()

    path_to_save=str(patches_dir) + "/"+str(zoom_level)+ "_crop.png"
    print(path_to_save)
    plt.imsave(path_to_save, thumbnail)

    ##full image with cords
    # img = np.array(thumbnail.convert('L'))
    # thresh = threshold_otsu(img)
    # print('threshold_otsu', thresh)
    # binary = img > thresh
    # f, axes = plt.subplots(1, 3, figsize=(20, 10));
    # ax = axes.ravel();
    # ax[0].imshow(img, cmap='gray');
    # ax[0].set_title('Original');
    # ax[1].hist(img.ravel(), bins=256);
    # ax[1].set_title('Histogram of pixel values');
    # ax[1].axvline(thresh, color='r');
    # ax[2].imshow(binary, cmap='gray');
    # ax[2].set_title('Binary');
    # f.savefig(str(BASE_TRUTH_DIR) + '/binary.png', dpi=f.dpi)

    sys.exit();

    thumbnail = np.array(thumbnail)
    patches = image.extract_patches_2d(thumbnail, (256, 256), max_patches=20000)
    print(patches.shape)
    patches_dir = str(BASE_TRUTH_DIR) + str(exp_folder_name) + '/'
    print('patches_dir', patches_dir)
    assure_path_exists(patches_dir)
    for counter, i in enumerate(patches):
        if np.any(i):
            matlabimg.imsave(str(patches_dir) + str(counter) + '.png', i)

    return patches


def assure_path_exists(path):
    # print("selected path", path)
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        # print("making dir")
        try:
            os.makedirs(dir)
        except OSError as e:
            if e.errno != errno.EEXIS:
                raise


if __name__ == "__main__":
    load_data_region(SLIDE_NAME_EXT)
