import numpy as np
from pathlib import Path
import os, errno, sys
from sklearn.feature_extraction import image
import matplotlib.image as matlabimg
from scipy.misc import imread
from skimage import io, data, img_as_float
# from skimage.measure import compare_ssim as ssim
import cv2
import matplotlib.pyplot as plt
from skimage.measure import structural_similarity as ssim

BASE_TRUTH_DIR = Path('/home/tarek/Downloads/mitosis@20x/eval-256-varying-dataset/')
SLIDE_NAME = '0_0.png'


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def compare_images(imageA, imageB, title):
    # compute the mean squared error and structural similarity
    # index for the images
    # m = mse(imageA, imageB)
    s = ssim(imageA, imageB, gradient=False, multichannel=True)
    print("S IS ",s)
    m = 1.02
    # setup the figure
    fig = plt.figure(title)
    # plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
    plt.suptitle(" SSIM: %.2f" % (s))

    # show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap=plt.cm.gray)
    plt.axis("off")

    # show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB, cmap=plt.cm.gray)
    plt.axis("off")

    # show the images
    plt.show()


def calculateSSIM():
    original_name = str(BASE_TRUTH_DIR) + "/6400/" + str(SLIDE_NAME)
    print(original_name)
    original = cv2.imread(original_name)

    synthesized_name = str(BASE_TRUTH_DIR) + "/H/" + str(SLIDE_NAME)
    print(synthesized_name)
    synthesized = cv2.imread(synthesized_name)

    # convert the images to grayscale
    # original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    # synthesized = cv2.cvtColor(synthesized, cv2.COLOR_BGR2GRAY)

    # compare the images
    compare_images(original, synthesized, "Original vs. synthesized")


if __name__ == "__main__":
    calculateSSIM()
