from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import os, errno, cv2
from glob import glob
import tifffile as tiff
import matplotlib.image as mp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import data, img_as_float
from skimage.measure import compare_ssim as ssim

plt.switch_backend('agg')

patch_size = (256, 256)
global_counter = 0


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


def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        try:
            os.makedirs(dir)
        except OSError as e:
            if e.errno != errno.EEXIS:
                raise


def read_png(path):
    """
    Read an image to RGB uint8
    :param path:
    :return:
    """
    im = cv2.imread(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


def generate_Scanner_eval_images(dir, output_dir, tag):
    pattern = "*.png"
    images = []
    print(output_dir)
    assure_path_exists(output_dir);
    # read directory of images
    for _, _, _ in os.walk(dir):
        images.extend(glob(os.path.join(dir, pattern)))

    images.sort()
    images = images[0:10]

    for counter, img in enumerate(images):
        # print(img)
        img = read_png(img)
        img = crop_center(img, 256, 256)
        cv2.imwrite(output_dir + str(counter) + '.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    media_dir = "/home/tarek/Downloads/mitosis@20x"

    tag = 'A'
    dir = media_dir + '/full_images_registered/' + tag
    output_dir = media_dir + "/diff_dimensions/256x256/" + tag + "/"
    generate_Scanner_eval_images(dir, output_dir, tag)

    print("_______________")
    tag = 'H'
    dir = media_dir + '/full_images_registered/' + tag
    output_dir = media_dir + "/diff_dimensions/256x256/" + tag + "/"
    generate_Scanner_eval_images(dir, output_dir, tag)
