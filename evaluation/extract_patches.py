# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os.path as osp
import openslide
from pathlib import Path
from skimage.filters import threshold_otsu
from openslide.deepzoom import DeepZoomGenerator
import os, errno
from keras.utils.np_utils import to_categorical
import matplotlib.image as mpimg
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from matplotlib import cm
from keras.models import load_model
from sklearn.model_selection import StratifiedShuffleSplit
from keras.callbacks import ModelCheckpoint
import sys
from datetime import datetime

# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'
BASE_TRUTH_DIR = Path('/home/tarek/Downloads/')
MASK_TRUTH_DIR = Path('/home/tarek/Downloads/Mask/')
META_NAME = 'exp13_patches_slide_83/'
exp_folder_name = 'Tumor_083_Babak_Normalized_250_370'
BATCH_SIZE = 32
N_EPOCHS = 30
NUM_SAMPLES = 36000


# NUM_SAMPLES = 36859

def load_data(slide_name):  # train_it()

    slide_path = str(BASE_TRUTH_DIR / slide_name)
    return find_patches_from_slide(slide_path)


## Define network

def seg_net(samples):
    from keras.models import Sequential
    from keras.layers import Lambda, Dropout
    from keras.layers.convolutional import Convolution2D, Conv2DTranspose
    from keras.layers.pooling import MaxPooling2D

    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(256, 256, 3)))
    model.add(Convolution2D(100, (5, 5), strides=(2, 2), activation='elu', padding='same'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(200, (5, 5), strides=(2, 2), activation='elu', padding='same'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(300, (3, 3), activation='elu', padding='same'))
    model.add(Convolution2D(300, (3, 3), activation='elu', padding='same'))
    model.add(Dropout(0.1))
    model.add(Convolution2D(2, (1, 1)))  # this is called upscore layer for some reason?
    model.add(Conv2DTranspose(2, (31, 31), strides=(16, 16), activation='softmax', padding='same'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # samples = find_patches_from_slide(slide_path)
    samples = samples.sample(NUM_SAMPLES, random_state=42)

    samples.reset_index(drop=True, inplace=True)

    # split samples into train and validation set
    # use StratifiedShuffleSplit to ensure both sets have same proportions of tumor patches
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(samples, samples["is_tumor"]):
        train_samples = samples.loc[train_index]
        validation_samples = samples.loc[test_index]

        print("train_samples  SAMPLES ", train_samples.is_tumor.value_counts())
        print("validation_samples  SAMPLES ", validation_samples.is_tumor.value_counts())

    ## TODO Add checkpoint mechanism to save old model before generating new

    # checkpoint
    filepath = "weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    train_generator = gen_imgs(train_samples, BATCH_SIZE)
    validation_generator = gen_imgs(validation_samples, BATCH_SIZE)

    # Train model
    train_start_time = datetime.now()

    model.fit_generator(train_generator, np.ceil(len(train_samples) / BATCH_SIZE),
                        validation_data=validation_generator,
                        validation_steps=np.ceil(len(validation_samples) / BATCH_SIZE),
                        epochs=N_EPOCHS, callbacks=callbacks_list)

    train_end_time = datetime.now()
    print("Model training time: %.1f minutes" % ((train_end_time - train_start_time).seconds / 60,))
    model.save(str(datetime.now()) + str(META_NAME) + str('.h5'))


## Data generator
def find_patches_from_slide(slide_path, base_truth_dir=BASE_TRUTH_DIR, filter_non_tissue=True):
    slide_contains_tumor = osp.basename(slide_path).startswith('Tumor_') or osp.basename(slide_path).startswith(
        'tumor_') or osp.basename(slide_path).startswith(
        'Test_')
    # slide_contains_tumor = True

    with openslide.open_slide(slide_path) as slide:
        print("Original Slide thumbnail %dx%d" % slide.dimensions)
        print(slide.level_downsamples)
        thumbnail = slide.read_region((250, 370), 0, (1000 , 1000))
        # print("Original Slide read_region %dx%d" % slide_resized.size)

        # thumbnail = slide.get_thumbnail((slide.dimensions[0] / 256, slide.dimensions[1] / 256))
        # thumbnail = slide.get_thumbnail((200,200))
        # print("Original Slide thumbnail %dx%d" % thumbnail.size)


    thumbnail_grey = np.array(thumbnail.convert('L'))  # convert to grayscale
    plt.imshow(thumbnail_grey, cmap='gray');
    # plt.imshow(thumbnail_grey);
    plt.savefig(str(osp.basename(slide_path)) + '.png')

    thresh = threshold_otsu(thumbnail_grey)
    print("thresh is ",thresh)
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


## Data generator
def find_classifier_patches_from_slide(slide_path, base_truth_dir=BASE_TRUTH_DIR, filter_non_tissue=True):
    # global global_patch_samples;
    """Returns a dataframe of all patches in slide
    input: slide_path: path to WSI file
    output: samples: dataframe with the following columns:
        slide_path: path of slide
        is_tissue: sample contains tissue
        is_tumor: truth status of sample
        tile_loc: coordinates of samples in slide


    option: base_truth_dir: directory of truth slides
    option: filter_non_tissue: Remove samples no tissue detected
    """
    slide_contains_tumor = osp.basename(slide_path).startswith('Tumor_')
    slide_contains_tumor = True

    with openslide.open_slide(slide_path) as slide:
        thumbnail = slide.get_thumbnail((slide.dimensions[0] / 256, slide.dimensions[1] / 256))
        print("Original Slide thumbnail %dx%d" % thumbnail.size)

    thumbnail_grey = np.array(thumbnail.convert('L'))  # convert to grayscale

    thresh = threshold_otsu(thumbnail_grey)
    binary = thumbnail_grey > thresh

    patches = pd.DataFrame(pd.DataFrame(binary))
    # print(patches)

    print("-----stacked---")
    patches = pd.DataFrame(pd.DataFrame(binary).stack())

    # print(patches)

    # print(patches)

    patches['is_tissue'] = ~patches[0]
    patches.drop(0, axis=1, inplace=True)
    patches['slide_path'] = slide_path
    if slide_contains_tumor:
        truth_slide_path = MASK_TRUTH_DIR / osp.basename(slide_path).replace('.tif', '_Mask.tif')
        print("truth_slide_path ", truth_slide_path)
        with openslide.open_slide(str(truth_slide_path)) as truth:
            thumbnail_truth = truth.get_thumbnail((truth.dimensions[0] / 256, truth.dimensions[1] / 256))

        patches_y = pd.DataFrame(pd.DataFrame(np.array(thumbnail_truth.convert("L"))).stack())
        print("patches_y Slide thumbnail", patches_y.size)

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


def gen_imgs(samples, batch_size, patches_dir, base_truth_dir=BASE_TRUTH_DIR, shuffle=True, test_mode=False):
    patches_dir = str(BASE_TRUTH_DIR) + '/' + str(META_NAME) + '/'
    if (patches_dir):
        patches_dir = patches_dir

    """This function returns a generator that
    yields tuples of (
        X: tensor, float - [batch_size, 256, 256, 3]
        y: tensor, int32 - [batch_size, 256, 256, NUM_CLASSES]
    )


    input: samples: samples dataframe
    input: batch_size: The number of images to return for each pull
    output: yield (X_train, y_train): generator of X, y tensors

    option: base_truth_dir: path, directory of truth slides
    option: shuffle: bool, if True shuffle samples
    """
    global global_counter
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        if shuffle:
            samples = samples.sample(frac=1)  # shuffle samples

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples.iloc[offset:offset + batch_size]

            images = []
            masks = []
            for counter, batch_sample in batch_samples.iterrows():
                slide_contains_tumor = osp.basename(batch_sample.slide_path).startswith('Tumor_')

                with openslide.open_slide(batch_sample.slide_path) as slide:
                    tiles = DeepZoomGenerator(slide, tile_size=256, overlap=0, limit_bounds=False)

                    if (os.path.isfile(str(patches_dir) + str(batch_sample.tile_loc[::-1]) + '.png')):

                        img = cv2.imread(str(patches_dir) + str(batch_sample.tile_loc[::-1]) + '.png')
                        b, g, r = cv2.split(img)  # get b,g,r
                        img = cv2.merge([r, g, b])  # switch it to rgb
                    #                         print("image was loaded")
                    else:
                        img = tiles.get_tile(tiles.level_count - 1, batch_sample.tile_loc[::-1])
                        cv2.imwrite(str(patches_dir) + str(batch_sample.tile_loc[::-1]) + '.png',
                                    cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))

                # only load truth mask for tumor slides
                if slide_contains_tumor:
                    truth_slide_path = MASK_TRUTH_DIR / osp.basename(batch_sample.slide_path).replace('.tif',
                                                                                                      '_Mask.tif')
                    with openslide.open_slide(str(truth_slide_path)) as truth:
                        truth_tiles = DeepZoomGenerator(truth, tile_size=256, overlap=0, limit_bounds=False)
                        mask = truth_tiles.get_tile(truth_tiles.level_count - 1, batch_sample.tile_loc[::-1])
                        mask = (cv2.cvtColor(np.array(mask), cv2.COLOR_RGB2GRAY) > 0).astype(int)
                else:
                    mask = np.zeros((256, 256))

                #                 cv2.imwrite('patches/'+str(batch_sample.tile_loc[::-1]) +'.png',cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
                #                 cv2.imwrite('patches/'+str(global_counter) +'.png',cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
                #                 cv2.imwrite('masks/'+str(global_counter) +'.png', mask)
                #                 global_counter=global_counter+1
                images.append(np.array(img))
                masks.append(mask)

            X_train = np.array(images)
            y_train = np.array(masks)
            y_train = to_categorical(y_train, num_classes=2).reshape(y_train.shape[0], 256, 256, 2)
            yield X_train, y_train


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


### Calculate Perfomance Metrics


def predict_batch_from_model(patches, model):
    """Predict which pixels are tumor.

    input: patch: `batch_size`x256x256x3, rgb image
    input: model: keras model
    output: prediction: 256x256x1, per-pixel tumor probability
    """
    predictions = model.predict(patches)
    predictions = predictions[:, :, :, 1]
    return predictions


def get_confusion_matrix(samples, model):
    # split samples into train and validation set
    # use StratifiedShuffleSplit to ensure both sets have same proportions of tumor patches
    # split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    # for train_index, test_index in split.split(samples, samples["is_tumor"]):
    #     train_samples = samples.loc[train_index]
    #     validation_samples = samples.loc[test_index]
    print('get_confusion_matrix Total patches in samples: %d' % len(samples))
    print('get_confusion_matrix Total Tumors in samples: ', samples.is_tumor.value_counts())

    # print("get_confusion_matrix train_samples  SAMPLES ", train_samples.is_tumor.value_counts())
    # print("get_confusion_matrix validation_samples  SAMPLES ", validation_samples.is_tumor.value_counts())

    # patches_dir = str(BASE_TRUTH_DIR) + str(META_NAME) + '/exp10-epoch30-slide9/'
    patches_dir = str(BASE_TRUTH_DIR) + str(META_NAME)
    # print("diiiiiiiir ",patches_dir)
    assure_path_exists(patches_dir + '/')
    validation_generator = gen_imgs(samples, BATCH_SIZE, patches_dir)
    validation_steps = np.ceil(len(samples) / BATCH_SIZE)

    confusion_mtx = np.zeros((2, 2))

    for i in tqdm(range(int(validation_steps))):
        X, y = next(validation_generator)
        preds = predict_batch_from_model(X, model)
        # print("PREDS ARE ",preds)
        y_true = y[:, :, :, 1].ravel()
        y_pred = np.uint8(preds > 0.5).ravel()
        # print("y_pred size ", y_predsize())
        # print("y_pred ARE ", y_pred)

        confusion_mtx += confusion_matrix(y_true, y_pred, labels=[0, 1])
    return confusion_mtx


def train_it():
    global_patch_samples = pd.concat([
        # load_data('Train_Normal/Normal_001.tif'),
        load_data('Train_Normal/Normal_006.tif'),
        # load_data('Train_Normal/Normal_011.tif'),
        # # load_data('Train_Normal/Normal_016.tif'),
        load_data('Train_Normal/Normal_034.tif'),

        load_data('Train_Tumor/Tumor_001.tif'),
        # load_data('Train_Tumor/Tumor_006.tif'),
        load_data('Train_Tumor/Tumor_009.tif'),
        # load_data('Train_Tumor/Tumor_010.tif'),
        load_data('Train_Tumor/Tumor_018.tif'),
        load_data('Train_Tumor/Tumor_025.tif'),
        load_data('Train_Tumor/Tumor_028.tif'),
        # load_data('Train_Tumor/Tumor_035.tif'),

    ], ignore_index=True);

    # global_patch_samples = pd.concat([
    #
    #     load_data('Train_Tumor/Tumor_009.tif'),
    #
    # ], ignore_index=True);

    # global_patch_samples = load_data('Train_Tumor/Tumor_009.tif');
    # print("-----------------")
    # print(global_patch_samples.columns)
    print('Total patches in global_patch_samples: %d' % len(global_patch_samples))
    print(global_patch_samples.is_tumor.value_counts())
    print(global_patch_samples.iloc[:5])

    enhanced_patches = global_patch_samples.copy()
    # delete false patches from original samples
    global_patch_samples = global_patch_samples[global_patch_samples['is_tumor'] == True]
    print('Total patches in global_patch_samples: %d' % len(global_patch_samples))

    print("----")
    enhanced_patches = enhanced_patches[enhanced_patches['is_tumor'] == False]
    enhanced_patches.sample(frac=1)
    enhanced_patches = enhanced_patches[:-161329]
    print('Total patches in enhanced_patches: %d' % len(enhanced_patches))
    print(enhanced_patches.is_tumor.value_counts())

    global_patch_samples = global_patch_samples.append(enhanced_patches, ignore_index=True)
    print('Total patches in global_patch_samples: %d' % len(global_patch_samples))
    print(global_patch_samples.is_tumor.value_counts())
    global_patch_samples.sample(frac=1)

    print(global_patch_samples.iloc[:5])

    seg_net(global_patch_samples)


def split_to_train_test(df, label_column, train_frac=0.8):
    train_df, test_df = pd.DataFrame(), pd.DataFrame()
    labels = df[label_column].unique()
    for lbl in labels:
        lbl_df = df[df[label_column] == lbl]
        lbl_train_df = lbl_df.sample(frac=train_frac)
        lbl_test_df = lbl_df.drop(lbl_train_df.index)
        # print('\n%s:\n---------\ntotal:%d\ntrain_df:%d\ntest_df:%d' % (
        # lbl, len(lbl_df), len(lbl_train_df), len(lbl_test_df)))
        train_df = train_df.append(lbl_train_df)
        test_df = test_df.append(lbl_test_df)

    return train_df, test_df


def screw_it(samples):
    # split samples into train and validation set
    # split = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
    # for train_index, test_index in split.split(samples, samples["is_tumor"]):
    #     train_samples = samples.loc[train_index]
    #     validation_test_samples = samples.loc[test_index]
    #
    # print("+++++++++++++++++++++++++++++++++++++++++++++++")
    # print("Train_samples ", train_samples.is_tumor.value_counts())
    # print("+++++++++++++++++++++++++++++++++++++++++++++++")
    #
    # print("validation_test_samples ", validation_test_samples.is_tumor.value_counts())
    # print("+++++++++++++++++++++++++++++++++++++++++++++++")
    # #
    # test_samples, validation_samples = split_to_train_test(validation_test_samples, 'is_tumor', 0.5)
    #
    # print("+++++++++++++++++++++++++++++++++++++++++++++++")
    # print("Test_samples ", test_samples.is_tumor.value_counts())
    # print("+++++++++++++++++++++++++++++++++++++++++++++++")
    # print("Validation_samples ", validation_samples.is_tumor.value_counts())
    # print("+++++++++++++++++++++++++++++++++++++++++++++++")

    # train patches
    exp_name = datetime.now()
    exp_name = "REALPATCHES"
    patches_dir = str(BASE_TRUTH_DIR) + '/patches/' + str(exp_name) + '/'+str(exp_folder_name)+'/'
    print('patches_dir', patches_dir)
    print('patches_dir', exp_folder_name)
    assure_path_exists(patches_dir)
    assure_path_exists(patches_dir + 'normal/')
    assure_path_exists(patches_dir + 'tumor/')
    assure_path_exists(patches_dir + 'mask/')


    # gen_imgs_classifier(validation_test_samples, patches_dir)
    gen_imgs_classifier(samples, patches_dir)
    #
    # # test patches
    # patches_dir = str(BASE_TRUTH_DIR) + '/patches/' + str(exp_name) + '/test/'
    # print('patches_dir', patches_dir)
    # assure_path_exists(patches_dir)
    # assure_path_exists(patches_dir + 'normal/')
    # assure_path_exists(patches_dir + 'tumor/')
    # assure_path_exists(patches_dir + 'mask/')
    # gen_imgs_classifier(test_samples, patches_dir)
    #
    # # validation patches
    # patches_dir = str(BASE_TRUTH_DIR) + '/patches/' + str(exp_name) + '/validation/'
    # print('patches_dir', patches_dir)
    # assure_path_exists(patches_dir)
    # assure_path_exists(patches_dir + 'normal/')
    # assure_path_exists(patches_dir + 'tumor/')
    # assure_path_exists(patches_dir + 'mask/')
    # gen_imgs_classifier(validation_samples, patches_dir)


def get_more_patches():
    # this time we wanna generrate for center 2
    # Center 2 tumor slides 071 - 110 and normal slides 101 - 160

    global_patch_samples = pd.concat([

        load_data('Train_Tumor/Tumor_083_Babak_Normalized.tif'),
        # load_data('Train_Tumor/Tumor_083.tif'),

    ], ignore_index=True);

    print('Total patches in global_patch_samples: %d' % len(global_patch_samples))
    print(global_patch_samples.is_tumor.value_counts())
    # global_patch_samples = global_patch_samples.sample(frac=1)

    enhanced_patches = global_patch_samples.copy()
    # delete false patches from original samples
    global_patch_samples = global_patch_samples[global_patch_samples['is_tumor'] == True]
    print('Total patches in global_patch_samples: %d' % len(global_patch_samples))

    print("----------")
    enhanced_patches = enhanced_patches[enhanced_patches['is_tumor'] == False]
    # enhanced_patches = enhanced_patches.sample(frac=1)
    # # 451947-40k
    # enhanced_patches = enhanced_patches[:-65839]
    # global_patch_samples = global_patch_samples.append(enhanced_patches, ignore_index=True)
    # global_patch_samples = global_patch_samples.sample(frac=1)
    # print(global_patch_samples.is_tumor.value_counts())
    print("----------")
    screw_it(global_patch_samples)


def test_it():
    # test_slide_name = 'Train_Tumor/Tumor_029.tif'
    test_slide_name = 'Testset_Part1/Test_016.tif'
    test_slide_path = str(BASE_TRUTH_DIR / test_slide_name)
    # model = load_model('/home/tarek/deployed/pytorch-cycle/evaluation/weights.best.hdf5')

    # Confusion matrix
    samples = load_data(test_slide_name)
    # print(samples.tail(5))
    # print('get_confusion_matrix Total patches in samples: %d' % len(samples))
    # print('get_confusion_matrix Total Tumors in samples: ', samples.is_tumor.value_counts())

    confusion_mtx = get_confusion_matrix(load_data(test_slide_name), model)
    print(confusion_mtx)

    tn = confusion_mtx[0, 0]
    fp = confusion_mtx[0, 1]
    fn = confusion_mtx[1, 0]
    tp = confusion_mtx[1, 1]
    print("tn ", tn)
    print("fp ", fp)
    print("fn ", fn)
    print("tp ", tp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1_score = 2 * ((precision * recall) / (precision + recall))

    print("Accuracy: %.2f" % accuracy)
    print("Recall: %.2f" % recall)
    print("Precision: %.2f" % precision)
    print("F1 Score: %.2f" % f1_score)


def main():
    # train_it()
    # screw_it()
    # get_more_patches()

if __name__ == "__main__":
    # print("openslide.__library_version__", openslide.__library_version__)
    main()
