import time
import os, sys
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer_time import Visualizer
from util import html

opt = TestOptions().parse()
opt.nThreads = 1  # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
# create website
# web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))

web_dir = os.path.join(opt.dataroot, '_StainGAN')
if opt.results_dir:
    web_dir = opt.results_dir

# print("web_dir ", web_dir)
# sys.exit()
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test
print("Dataset", len(dataset))
start_time = time.time()

for i, data in enumerate(dataset):
    if opt.how_many:
        print("how_many", opt.how_many)

        if i >= opt.how_many:
            break
    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals()
    img_path = model.get_image_paths()
    # print('%04d: process image... %s' % (i, img_path))
    print('%04d: process image' % (i))
    visualizer.save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio)

elapsed = (time.time() - start_time)
print("--- %s seconds ---" % round((elapsed ), 2))

# webpage.save()
