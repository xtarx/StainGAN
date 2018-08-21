import numpy as np
from pathlib import Path
import os, errno, sys
from sklearn.feature_extraction import image
import matplotlib.image as matlabimg
from scipy.misc import imread
from skimage import io

BASE_TRUTH_DIR = Path('/home/tarek/Downloads/mitosis@20x/png/')
SLIDE_NAME = 'A03_00B'
SLIDE_NAME_EXT = str(SLIDE_NAME) + '.png'
exp_folder_name = '/' + str(SLIDE_NAME) + '/'


def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        try:
            os.makedirs(dir)
        except OSError as e:
            if e.errno != errno.EEXIS:
                raise


# Function for obtaining center crops from an image
def crop_center(x, crop_w, crop_h):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    # print(h, w)
    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))
    # print(j, i)
    return x[j:j + crop_h, i:i + crop_w]


def stainWSI(slide_name):
    slide_path = str(BASE_TRUTH_DIR / slide_name)
    patches_dir = str(BASE_TRUTH_DIR) + str(exp_folder_name)
    meta_dir = str(BASE_TRUTH_DIR) + "meta/"

    print('patches_dir', patches_dir)
    print('meta_dir', meta_dir)
    assure_path_exists(patches_dir)
    assure_path_exists(meta_dir)

    # read the image
    img = imread(slide_path, mode='RGB')
    print("original image dimensions", img.shape)
    # resize

    img = crop_center(img, 1376, 1376)

    print("resized image dimensions", img.shape)

    # matlabimg.imsave(str(meta_dir) + 'IMG-CROPPED.png', img)
    matlabimg.imsave(str(meta_dir) + 'IMG-CROPPED_1376_1376.png', img)

    # img_array = fromimage(img, flatten=True)
    # print("img_array  dimensions", img_array.shape)

    sys.exit()
    # first split the image to 256x256 patches
    patches = image.extract_patches_2d(img, (32, 32))
    print("patches dimensions", patches.shape)

    for i in range(len(patches)):
        patch_idx = "_" + "%05d" % (i)
        patch_name = str(patches_dir) + "image_0001" + patch_idx + ".png"
        io.imsave(patch_name, patches[i])

    # reconstruct from directory

    patch_dir = str(patches_dir) + "*.png"
    patches = np.array(io.imread_collection(patch_dir))
    reconstructed = (image.reconstruct_from_patches_2d(patches, img.shape))

    print("reconstructed image dimensions", reconstructed.shape)
    import scipy.misc
    scipy.misc.imsave(str(meta_dir) + "reconstructed.png", img)
    print(np.testing.assert_array_equal(img, reconstructed))


if __name__ == "__main__":
    stainWSI(SLIDE_NAME_EXT)
