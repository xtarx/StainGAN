from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os, errno, cv2
from glob import glob
import matplotlib.pyplot as plt
from shutil import copyfile

plt.switch_backend('agg')

patch_size = (256, 256)
global_counter = 0


def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        try:
            os.makedirs(dir)
        except OSError as e:
            if e.errno != errno.EEXIS:
                raise


def generate_Scanner_eval_images(dir):
    pattern = "*.png"
    # print(dir)

    unstained = media_dir + '/unstained/normal/'
    output_dir1 = media_dir + '/tamam/Unnormalized/normal/'
    assure_path_exists(output_dir1);

    # babak = media_dir + '/babak/tumor/'
    # output_dir1 = media_dir + '/tamam/babak/tumor/'
    # assure_path_exists(output_dir1);
    #
    # Macenko = media_dir + '/Macenko/tumor/'
    # output_dir2 = media_dir + '/tamam/Macenko/tumor/'
    # assure_path_exists(output_dir2);
    #
    # Reinhard = media_dir + '/Reinhard/tumor/'
    # output_dir3 = media_dir + '/tamam/Reinhard/tumor/'
    # assure_path_exists(output_dir3);
    #
    # staingan = media_dir + '/staingan/tumor/'
    # output_dir4 = media_dir + '/tamam/staingan/tumor/'
    # assure_path_exists(output_dir4);
    #
    # vahadane = media_dir + '/vahadane/tumor/'
    # output_dir5 = media_dir + '/tamam/vahadane/tumor/'
    # assure_path_exists(output_dir5);

    # read directory of images
    images = []
    for _, _, _ in os.walk(dir):
        images.extend(glob(os.path.join(dir, pattern)))

    # print(images)
    images.sort()
    # images = images[0:10]

    for counter, img in enumerate(images):
        img_name = (img.split('/')[-1])
        # print(img_name)
        # copyfile(babak + img_name, output_dir1 + img_name)
        copyfile(unstained + img_name, output_dir1 + img_name)
        # copyfile(Macenko + img_name, output_dir2 + img_name)
        # copyfile(Reinhard + img_name, output_dir3 + img_name)
        # copyfile(staingan + img_name, output_dir4 + img_name)
        # copyfile(vahadane + img_name, output_dir5 + img_name)


def convert_to_jpg(dir):
    from PIL import Image

    pattern = "*.png"
    output_dir = media_dir + '/A_jpg/'
    assure_path_exists(output_dir);
    images = []
    for _, _, _ in os.walk(dir):
        images.extend(glob(os.path.join(dir, pattern)))
    # images = images[0:10]

    for counter, img in enumerate(images):
        img_name = (img.split('/')[-1])
        img_name = (img_name.split('.')[0])

        im = Image.open(img)
        rgb_im = im.convert('RGB')
        rgb_im.save(output_dir+img_name+'.jpg',quality=95)




if __name__ == '__main__':
    # media_dir = '/home/tarek/Downloads/patches/REALPATCHES/Babak_Exp/dataset'
    # media_dir = '/home/tarek/deployed/lua/CycleGAN/datasets/mitosis_dense_32'
    media_dir = '/home/tarek/Downloads/mitosis@20x/eval-256-varying-dataset'

    # inp_dir = media_dir + '/Khan/normal/'
    # generate_Scanner_eval_images(inp_dir)
    #
    inp_dir = media_dir + '/A/'
    convert_to_jpg(inp_dir)
