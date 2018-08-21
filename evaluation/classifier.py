import os

import errno
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras import applications
from keras.models import Model, load_model
from keras.utils.vis_utils import plot_model
import time

img_width, img_height = 256, 256
epochs = 30
train_samples = 29408
validation_samples = 10144
# validation_samples = 10816
test_samples = 10816
# test_samples = 6240
batch_size = 32
# exp_url = '/home/tarek/Downloads/patches/REALPATCHES/'
# /home/tarek/Downloads/patches/REALPATCHES/Babak_Exp/dataset
exp_url = '/home/tarek/Downloads/patches/REALPATCHES/Babak_Exp/dataset/tamam'
train_data_dir = str(exp_url) + 'train'
validation_data_dir = str(exp_url) + 'validation'
# test slides from center2---unstained

# test_data_dir = str(exp_url) + 'validation'
# test_data_dir = str(exp_url) + 'mini/test_center2_unstained'
# test_data_dir = str(exp_url) + 'mini/test_center2_stained_Mackenko'
# test_data_dir = str(exp_url) + 'mini/test_center2_stained_Reinhard'
# test_data_dir = str(exp_url) + 'mini/test_center2_stained_Khan'
test_data_dir = str(exp_url) + 'mini/test_center2_stained_Vahadane'


# test_data_dir = str(exp_url) + 'mini/test_center2_stained_GAN'


def create_bottleneck():
    # used to rescale the pixel values from [0, 255] to [0, 1] interval
    datagen = ImageDataGenerator(rescale=1. / 255)

    # VGG16 model is available in Keras

    model_vgg = applications.VGG16(include_top=False, weights='imagenet')

    # Using the VGG16 model to process samples

    train_generator_bottleneck = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    validation_generator_bottleneck = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    # This is a long process, so we save the output of the VGG16 once and for all.

    bottleneck_features_train = model_vgg.predict_generator(train_generator_bottleneck, train_samples // batch_size)
    np.save(open(str(exp_url) + 'models/bottleneck_features_train.npy', 'wb'), bottleneck_features_train)

    bottleneck_features_validation = model_vgg.predict_generator(validation_generator_bottleneck,
                                                                 validation_samples // batch_size)
    np.save(open(str(exp_url) + 'models/bottleneck_features_validation.npy', 'wb'), bottleneck_features_validation)


# Now we can load it...

def load_bottleneck():
    train_data = np.load(open(str(exp_url) + 'models/bottleneck_features_train.npy', 'rb'))
    train_labels = np.array([0] * (train_samples // 2) + [1] * (train_samples // 2))

    validation_data = np.load(open(str(exp_url) + 'models/bottleneck_features_validation.npy', 'rb'))
    validation_labels = np.array([0] * (validation_samples // 2) + [1] * (validation_samples // 2))

    print('train_data.shape', train_data.shape)
    print('train_labels', train_labels.shape)

    print('validation_data.shape', validation_data.shape)
    print('validation_labels', validation_labels.shape)
    # And define and train the custom fully connected neural network :

    model_top = Sequential()
    model_top.add(Flatten(input_shape=train_data.shape[1:]))
    model_top.add(Dense(256, activation='relu'))
    model_top.add(Dropout(0.5))
    model_top.add(Dense(1, activation='sigmoid'))

    model_top.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    model_top.fit(train_data, train_labels,
                  epochs=epochs,
                  batch_size=batch_size,
                  validation_data=(validation_data, validation_labels))

    # The training process of this small neural network is very fast : ~2s per epoch

    model_top.save_weights(str(exp_url) + 'models/bottleneck_30_epochs.h5')
    # model_top.load_weights(str(exp_url) + 'models/bottleneck_30_epochs.h5')

    # ### Bottleneck model evaluation
    print(model_top.evaluate(validation_data, validation_labels))


def fine_tune():
    # Start by instantiating the VGG base and loading its weights.
    epochs = 10

    model_vgg = applications.VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

    # Build a classifier model to put on top of the convolutional model. For the fine tuning, we start with a fully trained-classifer. We will use the weights from the earlier model. And then we will add this model on top of the convolutional base.

    top_model = Sequential()
    top_model.add(Flatten(input_shape=model_vgg.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid'))

    top_model.load_weights(str(exp_url) + 'models/bottleneck_30_epochs.h5')

    # model_vgg.add(top_model)
    model = Model(inputs=model_vgg.input, outputs=top_model(model_vgg.output))

    # For fine turning, we only want to train a few layers.  This line will set the first 25 layers (up to the conv block) to non-trainable.

    for layer in model.layers[:15]:
        layer.trainable = False

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    # prepare data augmentation configuration  . . . do we need this?
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary')

    # fine-tune the model
    model.fit_generator(
        train_generator,
        steps_per_epoch=train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_samples // batch_size)

    model.save_weights(str(exp_url) + 'models/finetuning_30epochs_vgg.h5')
    model.save(str(exp_url) + 'models/theultimate.h5')

    # ### Evaluating on validation set

    # Computing loss and accuracy :

    print(model.evaluate_generator(validation_generator, validation_samples))


def test2():
    model = load_model(str(exp_url) + 'models/theultimate.h5')

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary')

    print(model.metrics_names)
    # Computing loss and accuracy :

    print(model.evaluate_generator(test_generator, test_samples // batch_size))


def test():
    print("TESTING USING: ", test_data_dir)
    model = load_model(str(exp_url) + 'models/theultimate.h5')

    datagen = ImageDataGenerator(rescale=1. / 255)

    # get confision matrix
    generator = datagen.flow_from_directory(
        test_data_dir,
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode=None,  # only data, no labels
        # class_mode='binary',  # only data, no labels
        shuffle=False)  # keep data in same order as labels
    print(generator.class_indices)

    probabilities = model.predict_generator(generator, test_samples // batch_size)
    from sklearn.metrics import confusion_matrix

    # print(probabilities[10])
    y_true = np.array([0] * (test_samples // 2) + [1] * (test_samples // 2))
    y_pred = probabilities > 0.8
    print(y_pred, y_pred)
    confusion_mtx = (confusion_matrix(y_true, y_pred))
    print(confusion_mtx)
    tn = confusion_mtx[0, 0]
    fp = confusion_mtx[0, 1]
    fn = confusion_mtx[1, 0]
    tp = confusion_mtx[1, 1]
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1_score = 2 * ((precision * recall) / (precision + recall))

    print("Accuracy: %.2f" % accuracy)
    print("Recall: %.2f" % recall)
    print("Precision: %.2f" % precision)
    print("F1 Score: %.2f" % f1_score)


def test3():
    # test_samples = 640
    print("TESTING USING: ", test_data_dir)
    model = load_model(str(exp_url) + 'models/theultimate.h5')

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])
    datagen = ImageDataGenerator(rescale=1. / 255)

    # get confision matrix
    generator = datagen.flow_from_directory(
        test_data_dir,
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True)  # keep data in same order as labels
    print(generator.class_indices)

    score = model.evaluate_generator(generator, test_samples // batch_size)

    print("score", score)
    scores = model.predict_generator(generator, test_samples // batch_size)
    print(scores.shape)
    print(scores)
    correct = 0
    for i, n in enumerate(generator.filenames):
        if i < test_samples:
            # print(i)
            if n.startswith("normal") and scores[i] <= 0.5:
                correct += 1
            if n.startswith("tumor") and scores[i] > 0.5:
                # print("TUMOOOOOOOOOOOR")
                correct += 1

    print("Correct:", correct, " Total: ", len(generator.filenames))
    print("Loss: ", score[0], "Accuracy: ", score[1])


def cnn():
    ##preprocessing
    # used to rescale the pixel values from [0, 255] to [0, 1] interval
    datagen = ImageDataGenerator(rescale=1. / 255)
    # batch_size = 32

    # automagically retrieve images and their classes for train and validation sets
    train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    validation_generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')
    from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D

    # a simple stack of 3 convolution layers with a ReLU activation and followed by max-pooling layers.
    model = Sequential()
    model.add(Convolution2D(32, (3, 3), input_shape=(img_width, img_height, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    model.fit_generator(
        train_generator,
        steps_per_epoch=train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_samples // batch_size, )

    model.save_weights(str(exp_url) + 'models/basic_cnn_30_epochs_weights.h5')

    model.save(str(exp_url) + 'models/basic_cnn_30_epochs_full_model.h5')


def cnn_plot():
    from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D

    # a simple stack of 3 convolution layers with a ReLU activation and followed by max-pooling layers.
    model = Sequential()
    model.add(Convolution2D(32, (3, 3), input_shape=(img_width, img_height, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    plot_model(model, to_file=str(exp_url) + 'models/model_plot.png', show_shapes=True, show_layer_names=True)


def plot_roc_curve(fpr, tpr, roc_auc, name=''):
    import matplotlib.pyplot as plt

    print("Producing Chart - ROC Curve - {}".format(name))
    plt.figure()
    plt.plot(fpr,
             tpr,
             label='ROC curve (AUC = %0.2f)' % roc_auc)

    # plt.plot(fpr,
    #          fpr,
    #          label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1],
             [0, 1],
             'k--')

    plt.xlim([0.0,
              1.0])
    plt.ylim([0.0,
              1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(str(exp_url) + 'plots/' + name + '_ROC_Curve.png',
                bbox_inches='tight')


def plot_roc_curve_multiple(fpr, tpr, roc_auc, names, exp_url):
    import matplotlib.pyplot as plt

    print("Producing Chart - ROC Curve - {}".format('all in one'))
    plt.figure()

    # for fp, tp, roc, name in fpr, tpr, roc_auc, names:
    print(len(names))
    for i in range(len(names)):
        plt.plot(np.array(fpr[i]),
                 np.array(tpr[i]),
                 label=str(names[i]) + ' (AUC = %0.2f)' % roc_auc[i])
    # plt.plot(fpr,
    #          fpr,
    #          label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1],
             [0, 1],
             'k--')

    plt.xlim([0.0,
              1.0])
    plt.ylim([0.0,
              1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(str(exp_url) + 'plots/' + str(int(time.time())) + 'TAMAM_all_in_1_ROC_Curve.png',
                bbox_inches='tight')


def cnn_test(methods):
    # model = load_model(str(exp_url) + 'models/basic_cnn_30_epochs_full_model.h5')
    model = load_model('/home/tarek/Downloads/patches/REALPATCHES/models/basic_cnn_30_epochs_full_model.h5')

    fpr_arr = []
    tpr_arr = []
    roc_arr = []
    names_arr = []
    from sklearn.metrics import roc_auc_score, roc_curve, auc

    for method in methods:
        # method_test_dir = str(exp_url) + 'mini/' + str(method)
        method_test_dir = str(exp_url) + '/' + str(method)
        test_generator = ImageDataGenerator(rescale=1. / 255, rotation_range=20,
                                            width_shift_range=0.2,
                                            height_shift_range=0.2).flow_from_directory(
            method_test_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            shuffle=False,

            class_mode='binary')
        # test_generator.samples=3000
        no_samples = test_generator.samples
        # no_samples =3000
        test_steps = np.ceil(no_samples / test_generator.batch_size)

        true_classes = test_generator.classes

        score = model.evaluate_generator(test_generator, test_steps)

        print("score", score)

        predictions = model.predict_generator(generator=test_generator, steps=test_steps)
        auc_score = roc_auc_score(y_true=true_classes, y_score=predictions)

        method_name_beautify = (method.split('test_center2_')[-1])
        print("-----------------------------------------")
        print("cnn_test USING: ", method_name_beautify)

        print('auc score from generator', auc_score)
        print('true_classes ', true_classes.shape)
        print('y_score ', predictions.shape)
        fpr, tpr, _ = roc_curve(true_classes, predictions)
        print('fpr ', fpr.shape)
        print('tpr ', tpr.shape)
        roc_auc = auc(fpr, tpr)
        print("roc auc", roc_auc)
        print("-----------------------------------------")
        fpr_arr.append(fpr)
        tpr_arr.append(tpr)
        roc_arr.append(roc_auc)
        names_arr.append(method_name_beautify)

    plot_roc_curve_multiple(fpr_arr, tpr_arr, roc_arr, names_arr, '/home/tarek/Downloads/patches/REALPATCHES/')


def assure_path_exists(path):
    print("selected path", path)
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        # print("making dir")
        os.makedirs(dir)


def copy_files_to_dir(directory, which_class):
    from shutil import copy
    from glob import glob

    pattern = "*.png"
    images = []
    directory = str(exp_url) + directory + str(which_class)
    print(directory)

    for _, _, _ in os.walk(directory):
        images.extend(glob(os.path.join(directory, pattern)))

    print(len(images))
    images.sort()

    assure_path_exists(str(exp_url) + 'mini/test_center2_unstained/' + str(which_class))
    assure_path_exists(str(exp_url) + 'mini/test_center2_stained_GAN/' + str(which_class))
    assure_path_exists(str(exp_url) + 'mini/test_center2_stained_Mackenko/' + str(which_class))
    assure_path_exists(str(exp_url) + 'mini/test_center2_stained_Reinhard/' + str(which_class))
    assure_path_exists(str(exp_url) + 'mini/test_center2_stained_Khan/' + str(which_class))
    assure_path_exists(str(exp_url) + 'mini/test_center2_stained_Vahadane/' + str(which_class))
    for img in images:
        img_name = (img.split('/')[-1])
        print("copying ", img_name)

        copy(str(exp_url) + 'test_center2_unstained/' + str(which_class) + img_name,
             str(exp_url) + 'mini/test_center2_unstained/' + str(which_class) + img_name)

        copy(str(exp_url) + 'test_center2_stained_GAN/' + str(which_class) + img_name,
             str(exp_url) + 'mini/test_center2_stained_GAN/' + str(which_class) + img_name)

        copy(str(exp_url) + 'test_center2_stained_Mackenko/' + str(which_class) + img_name,
             str(exp_url) + 'mini/test_center2_stained_Mackenko/' + str(which_class) + img_name)

        copy(str(exp_url) + 'test_center2_stained_Reinhard/' + str(which_class) + img_name,
             str(exp_url) + 'mini/test_center2_stained_Reinhard/' + str(which_class) + img_name)

        copy(str(exp_url) + 'test_center2_stained_Vahadane/' + str(which_class) + img_name,
             str(exp_url) + 'mini/test_center2_stained_Vahadane/' + str(which_class) + img_name)


def dat_file():
    print("Adas")
    arr = np.fromfile('eval_ssim.dat')
    print(arr.shape)
    print(int(arr[0]))


def main():
    print(str(int(time.time())))
    # dat_file();
    # cnn_plot()
    # exit()
    # create_bottleneck()
    # load_bottleneck()
    # fine_tune()
    # test()
    # test3()
    # cnn()
    # cnn_test()
    # test_data_dir = str(exp_url) + 'validation'
    # test_data_dir = str(exp_url) + 'mini/test_center2_unstained'
    # test_data_dir = str(exp_url) + 'mini/test_center2_stained_Mackenko'
    # test_data_dir = str(exp_url) + 'mini/test_center2_stained_Reinhard'
    # test_data_dir = str(exp_url) + 'mini/test_center2_stained_Khan'
    #
    # cnn_test(['test_center2_unstained', 'test_center2_stained_Mackenko', 'test_center2_stained_Reinhard',
    #           'test_center2_stained_Khan',
    #           'test_center2_stained_Vahadane',
    #           'test_center2_stained_GAN']
    #          )
    # cnn_test(['test_center2_unstained', 'test_center2_stained_Mackenko','test_center2_stained_GAN'])
    # cnn_test([ 'test_center2_unstained'])
    # cnn_test(['Unnormalized', 'Reinhard', 'Mackenko', 'Khan', 'Vahadane', 'StainGAN'])
    cnn_test(['Unnormalized', 'Reinhard', 'Macenko', 'Khan', 'Bejnordi', 'Vahadane', 'StainGAN'])
    # copy_files_to_dir('mini/test_center2_stained_Khan/', 'tumor/')
    # copy_files_to_dir('mini/test_center2_stained_Khan/', 'normal/')


if __name__ == "__main__":
    main()
