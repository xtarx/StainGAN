import os
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras import applications
from keras.models import Model, load_model

img_width, img_height = 256, 256
epochs = 20
# 15762
train_samples = 40000
validation_samples = 13984
batch_size = 32
exp_url = '/home/tarek/Downloads/patches/expp2/'
train_data_dir = str(exp_url) + 'train'
validation_data_dir = str(exp_url) + 'validation'
test_data_dir = str(exp_url) + 'test'


def predict_from_model(patch, model):
    """Predict which pixels are tumor.

    input: patch: 256x256x3, rgb image
    input: model: keras model
    output: prediction: 256x256x1, per-pixel tumor probability
    """

    prediction = model.predict(patch.reshape(1, 256, 256, 3))
    prediction = prediction[:, :, :, 1].reshape(256, 256)
    return prediction


def predict_batch_from_model(patches, model):
    """Predict which pixels are tumor.

    input: patch: `batch_size`x256x256x3, rgb image
    input: model: keras model
    output: prediction: 256x256x1, per-pixel tumor probability
    """
    predictions = model.predict(patches)
    predictions = predictions[:, :, :, 1]
    return predictions


def build(source=None):
    datagen = ImageDataGenerator(rescale=1. / 255)
    data_generator = datagen.flow_from_directory(
        source,  # this is the target directory
        target_size=(256, 256),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='sparse')
    class_dictionary = data_generator.class_indices
    return data_generator, class_dictionary




def gen_imgs():

    global global_counter
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples.iloc[offset:offset + batch_size]

            images = []
            masks = []
            for counter, batch_sample in batch_samples.iterrows():

                np.array(img)

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

                images.append(np.array(img))
                masks.append(mask)

            X_train = np.array(images)
            y_train = np.array(masks)
            y_train = to_categorical(y_train, num_classes=2).reshape(y_train.shape[0], 256, 256, 2)
            yield X_train, y_train


# Data generator
def find_patches_from_slide(slide_path):
    import numpy as np
    import pandas as pd
    from glob import glob

    patches = pd.DataFrame()
    pattern = "*.png"
    images = []
    curr_dir=test_data_dir+'tumor/'
    for _, _, _ in os.walk(curr_dir):
        images.extend(glob(os.path.join(curr_dir, pattern)))

    print(len(images))
    images.sort()
    images = images[100:]
    print("First Index is ", images[0])
    for img in images:
        resized_images.append((tiff.imread(img)))

    if slide_contains_tumor:

        patches_y['is_tumor'] = patches_y[0] > 0
        patches_y.drop(0, axis=1, inplace=True)

        samples = pd.concat([patches, patches_y], axis=1)

    else:
        samples = patches
        samples['is_tumor'] = False

    samples.reset_index(inplace=True, drop=True)

    return samples


def main():
    build()


if __name__ == "__main__":
    main()
