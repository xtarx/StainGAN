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

BASE_TRUTH_DIR = Path('/home/tarek/Downloads/')
MASK_TRUTH_DIR = Path('/home/tarek/Downloads/Mask/')
exp_folder_name = 'Babak_Exp/Tumor_083_250_370'
exp_name = "REALPATCHES"


def load_data(slide_name):
    slide_path = str(BASE_TRUTH_DIR / slide_name)
    return find_patches_from_slide(slide_path)


## Data generator
def find_patches_from_slide(slide_path, filter_non_tissue=True):
    slide_contains_tumor = osp.basename(slide_path).startswith('Tumor_') or osp.basename(slide_path).startswith(
        'tumor_') or osp.basename(slide_path).startswith(
        'Test_')
    # slide_contains_tumor = True

    with openslide.open_slide(slide_path) as slide:
        print("Original Slide thumbnail %dx%d" % slide.dimensions)
        print(slide.level_downsamples)
        # thumbnail = slide.read_region((300, 25), 0, (1000, 1000))
        thumbnail = slide.read_region((250, 370), 0, (1000, 1000))
        # print("Original Slide read_region %dx%d" % slide_resized.size)

        # thumbnail = slide.get_thumbnail((slide.dimensions[0] / 256, slide.dimensions[1] / 256))
        # thumbnail = slide.get_thumbnail((200,200))
        # print("Original Slide thumbnail %dx%d" % thumbnail.size)

    thumbnail_grey = np.array(thumbnail.convert('L'))  # convert to grayscale
    plt.imshow(thumbnail_grey, cmap='gray');
    # plt.imshow(thumbnail_grey);
    plt.savefig(str(osp.basename(slide_path)) + '.png')

    thresh = threshold_otsu(thumbnail_grey)
    print("thresh is ", thresh)
    # sys.exit();

    binary = thumbnail_grey > thresh

    patches = pd.DataFrame(pd.DataFrame(binary).stack())

    patches['is_tissue'] = ~patches[0]
    patches.drop(0, axis=1, inplace=True)
    patches['slide_path'] = slide_path
    if slide_contains_tumor:
        truth_slide_path = MASK_TRUTH_DIR / osp.basename(slide_path).replace('.tif', '_Mask.tif')
        with openslide.open_slide(str(truth_slide_path)) as truth:
            thumbnail_truth = truth.get_thumbnail((truth.dimensions[0] / 256, truth.dimensions[1] / 256))
        patches_y = pd.DataFrame(pd.DataFrame(np.array(thumbnail_truth.convert("L"))).stack())

        patches_y['is_tumor'] = patches_y[0] > 0
        patches_y.drop(0, axis=1, inplace=True)

        samples = pd.concat([patches, patches_y], axis=1)

    else:
        samples = patches
        samples['is_tumor'] = False

    if filter_non_tissue:
        samples = samples[samples.is_tissue == True]  # remove patches with no tissue
    samples['tile_loc'] = list(samples.index)
    samples.reset_index(inplace=True, drop=True)

    return samples


def gen_imgs_classifier(samples, patches_dir):
    num_samples = len(samples)
    print("gen_imgs_classifier ", num_samples)

    for counter, batch_sample in samples.iterrows():

        with openslide.open_slide(batch_sample.slide_path) as slide:
            tiles = DeepZoomGenerator(slide, tile_size=256, overlap=0, limit_bounds=False)

            img = tiles.get_tile(tiles.level_count - 1, batch_sample.tile_loc[::-1])

        # only load truth mask for tumor slides
        if batch_sample.is_tumor:
            truth_slide_path = MASK_TRUTH_DIR / osp.basename(batch_sample.slide_path).replace('.tif',
                                                                                              '_Mask.tif')
            with openslide.open_slide(str(truth_slide_path)) as truth:
                truth_tiles = DeepZoomGenerator(truth, tile_size=256, overlap=0, limit_bounds=False)
                mask = truth_tiles.get_tile(truth_tiles.level_count - 1, batch_sample.tile_loc[::-1])
                # check center patch (128,128) if WHITE then mark as tumor
                mask_n = np.array(mask)
                mask_center = mask_n[128, 128]

                if mask_center[0] == 255:
                    # print("ITS A TUMOR !!!!!!")
                    cv2.imwrite(str(patches_dir) + 'mask/' + str(batch_sample.tile_loc[::-1]) + '.png',
                                cv2.cvtColor(np.array(mask_n), cv2.COLOR_RGB2BGR))
                    cv2.imwrite(str(patches_dir) + 'tumor/' + str(batch_sample.tile_loc[::-1]) + '.png',
                                cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))

        else:
            # normal
            cv2.imwrite(str(patches_dir) + 'normal/' + str(batch_sample.tile_loc[::-1]) + '.png',
                        cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))


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


def screw_it(samples):
    print('screw_it Total patches in global_patch_samples: %d' % len(samples))
    # exp_name = datetime.now()
    patches_dir = str(BASE_TRUTH_DIR) + '/patches/' + str(exp_name) + '/' + str(exp_folder_name) + '/'
    print('patches_dir', patches_dir)
    print('patches_dir', exp_folder_name)
    assure_path_exists(patches_dir)
    assure_path_exists(patches_dir + 'normal/')
    assure_path_exists(patches_dir + 'tumor/')
    assure_path_exists(patches_dir + 'mask/')
    gen_imgs_classifier(samples, patches_dir)


def get_more_patches():
    # this time we wanna generrate for center 2
    # Center 2 tumor slides 071 - 110 and normal slides 101 - 160
    global_patch_samples = pd.concat([
        # load_data('Train_Tumor/Tumor_083_Babak_Normalized.tif'),
        load_data('Train_Tumor/Tumor_083.tif'),
    ], ignore_index=True);

    print('Total patches in global_patch_samples: %d' % len(global_patch_samples))
    print(global_patch_samples.is_tumor.value_counts())
    # global_patch_samples = global_patch_samples.sample(frac=1)

    enhanced_patches = global_patch_samples.copy()
    # delete false patches from original samples
    global_patch_samples = global_patch_samples[global_patch_samples['is_tumor'] == True]
    print('Total patches in global_patch_samples: %d' % len(global_patch_samples))

    enhanced_patches = enhanced_patches[enhanced_patches['is_tumor'] == False]
    # enhanced_patches = enhanced_patches.sample(frac=1)
    # # 451947-40k
    enhanced_patches = enhanced_patches[:-65839]
    global_patch_samples = global_patch_samples.append(enhanced_patches, ignore_index=True)
    global_patch_samples = global_patch_samples.sample(frac=1)
    print(global_patch_samples.is_tumor.value_counts())
    print("----------")
    screw_it(global_patch_samples)


def remove_outliers():
    # sampl_outlier = '/home/tarek/Downloads/patches/REALPATCHES/Babak_Exp/Tumor_083_300_25/try3.png'
    # avg_color_per_row = np.average(cv2.imread(sampl_outlier, 0), axis=0)
    # avg_color = np.average(avg_color_per_row, axis=0)
    # print(avg_color)

    pattern = "*.png"
    images = []
    patches_dir = str(BASE_TRUTH_DIR) + '/patches/' + str(exp_name) + '/' + str(exp_folder_name) + '/'

    imgs_dir = patches_dir + 'normal/'
    imgs_new_dir = patches_dir + '/normal_filtered/'
    assure_path_exists(imgs_new_dir)
    for _, _, _ in os.walk(imgs_dir):
        images.extend(glob(os.path.join(imgs_dir, pattern)))

    print(len(images))
    images.sort()

    print("First Index is ", images[0])
    incl = 0
    for img in images:
        img_name = (img.split('/')[-1])
        avg_color_per_row = np.average(cv2.imread(img, 0), axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)

        if avg_color < 200:
            incl = incl + 1
            # print(avg_color)
            os.rename(img, imgs_new_dir + img_name)

    print("incl  ", incl)


def make_babak_dataset():
    pattern = "*.png"
    images = []

    slide_pos = '250_170'
    current_state = 'tumor'
    exp_folder_name = 'Babak_Exp/Tumor_083_Babak_Normalized_' + str(slide_pos) + '/' + str(current_state)
    babak_patches_dir = str(BASE_TRUTH_DIR) + '/patches/' + str(exp_name) + '/' + str(exp_folder_name) + '/'
    exp_folder_name = 'Babak_Exp/Tumor_083_' + str(slide_pos) + '/' + str(current_state)

    print(babak_patches_dir)
    unstained_patches_dir = str(BASE_TRUTH_DIR) + '/patches/' + str(exp_name) + '/' + str(exp_folder_name) + '/'
    from shutil import copyfile

    # assure_path_exists(imgs_new_dir)
    for _, _, _ in os.walk(babak_patches_dir):
        images.extend(glob(os.path.join(babak_patches_dir, pattern)))

    print(len(images))
    # images = images[0:10]
    images.sort()

    babak_new_dir = str(BASE_TRUTH_DIR) + '/patches/' + str(exp_name) + '/Babak_Exp/dataset/babak/' + str(
        current_state) + '/'
    unstained_new_dir = str(BASE_TRUTH_DIR) + '/patches/' + str(exp_name) + '/Babak_Exp/dataset/unstained/' + str(
        current_state) + '/'
    incl = 200
    for babak_img in images:
        img_name = (babak_img.split('/')[-1])

        unstianed_img_path = str(unstained_patches_dir + img_name)
        babak_img_path = str(babak_patches_dir + img_name)
        # print("babak path ",babak_img_path)
        # print("unstianed_img_path path ",unstianed_img_path)
        if os.path.exists(babak_img_path) and os.path.exists(unstianed_img_path):
            print("making dir")
            try:
                incl = incl + 1
                print("match #", incl)
                new_img_name = str(incl) + str('.png')
                copyfile(babak_img, babak_new_dir + new_img_name)
                copyfile(unstained_patches_dir + img_name, unstained_new_dir + new_img_name)

            except OSError as e:
                if e.errno != errno.EEXIS:
                    raise

    print("final incl is ", incl)


def test_gradient():
    p1 = '/home/tarek/deployed/pytorch-cycle/datasets/camelyon16/trainA/(20, 175).png'
    # img1 = np.array(cv2.imread(str(p1)))
    p2 = '/home/tarek/deployed/pytorch-cycle/datasets/camelyon16/trainA/(20, 191).png'
    # img2 = np.array(cv2.imread(str(p2)))

    # plotting
    # gx, gy = np.gradient(np.array(cv2.imread(str(p1))))
    gx, gy = np.gradient(np.array(cv2.imread(str(p1))))

    plt.close("all")
    plt.figure()
    plt.suptitle("Image, and it gradient along each axis")
    ax = plt.subplot("131")
    ax.axis("off")
    ax.imshow(p1)
    ax.set_title("image")

    ax = plt.subplot("132")
    ax.axis("off")
    ax.imshow(gx)
    ax.set_title("gx")

    ax = plt.subplot("133")
    ax.axis("off")
    ax.imshow(gy)
    ax.set_title("gy")
    # plt.show()
    plt.savefig('masss.png')
    
    #
    #
    # grad1 = get_gradient(p1)
    # cv2.imwrite(grad1)
    # grad2 = get_gradient(p2)
    #
    # print(grad1)
    #
    # print("grad 222222")
    # print(grad2)
    # grad2 = get_gradient(p2)
    #
    # wx=np.multiply(grad1, img1)
    # wg=np.multiply(grad1, img2)
    #
    # # print(wx)
    # # print(wg)
    # print(wx-wg)

    # np.multiply(a, b)


def get_gradient(path):
    return np.gradient(np.array(cv2.imread(str(path))))


if __name__ == "__main__":
    # make_babak_dataset()
    # get_more_patches()

    # remove_outliers()

    test_gradient()
