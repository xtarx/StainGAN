import time, errno, cv2, os
import stain_utils as utils
import stainNorm_Reinhard
import stainNorm_Macenko
import stainNorm_Vahadane
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

exp_name = ""
# media_url = "/home/tarek/Downloads/mitosis@20x/evaluate_noise/"
media_url = '/home/tarek/Downloads/patches/REALPATCHES/Babak_Exp/dataset/tamam/'

i1 = utils.read_image(media_url + exp_name + "/tumor18.png")


def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        try:
            os.makedirs(dir)
        except OSError as e:
            if e.errno != errno.EEXIS:
                raise


def transform_imgs(dir, output_dir, normalizer=''):
    pattern = "*.png"
    images = []
    # read directory of images
    for _, _, _ in os.walk(dir):
        images.extend(glob(os.path.join(dir, pattern)))
    # print(len(images))
    images.sort()
    # images = images[0:2]

    for counter, img in enumerate(images):
        img_name = (img.split('/')[-1])
        transformed_img = normalizer.transform(utils.read_image(img))
        cv2.imwrite(output_dir + str(img_name), cv2.cvtColor(transformed_img, cv2.COLOR_RGB2BGR))


# start working here

print(exp_name)
input_dir = media_url + "Unnormalized/tumor/"

output_dir = media_url + "/Reinhard/tumor/"
assure_path_exists(output_dir)
print(output_dir)

start_time = time.time()

n = stainNorm_Reinhard.normalizer()
n.fit(i1)
transform_imgs(input_dir, output_dir, n)

elapsed = (time.time() - start_time)
print("--- %s seconds ---" % round((elapsed / 2), 2))

#
# output_dir = media_url + exp_name + "/Vahadane/"
#
# assure_path_exists(output_dir)
# start_time = time.time()
#
# n = stainNorm_Vahadane.normalizer()
# n.fit(i1)
# transform_imgs(input_dir, output_dir, n)
#
# elapsed = (time.time() - start_time)
# print("--- %s seconds ---" % round((elapsed / 2), 2))

# n = stainNorm_Macenko.normalizer()
# n.fit(i1)
# output_dir = "/Users/xtarx/Documents/TUM/5th/Thesis/dataset/mitosis@20x/eval/macenko/"
# transform_imgs(input_dir, output_dir, n)
#
