import time, sys
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer_time import Visualizer

opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt)
total_steps = 0

# for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):

train_start_time = time.time()
no_saved = 1
frequency_count = 1

# REAL MODEL
train_end_time = train_start_time + (3600 * 4)
save_model_frequency = 3600
display_frequency = 300
print_frequency = 300

# TRAIL MODEL
# train_end_time = train_start_time + (180 * 4)
# save_model_frequency = 180
# display_frequency = 60
# print_frequency = 60
print(len(dataset))
while True:

    # print(epoch_start_time)
    # sys.exit()
    # epoch_iter = 0

    for i, data in enumerate(dataset):
        print("IN ITERATION ", i)
        iter_start_time = time.time()
        visualizer.reset()
        total_steps += opt.batchSize
        model.set_input(data)
        model.optimize_parameters()

        if time.time() >= train_end_time:
            model.save('latest-for-real')
            total_train_time = (train_end_time - train_start_time) / 3600
            print("End of Training  start time %s, end time %s - total time %s" % (
                train_start_time, train_end_time, total_train_time))
            print("EXITING TRAINING")
            sys.exit()
            break

        if time.time() >= train_start_time + (no_saved * save_model_frequency):
            print('saving the model at the end of no_saved %d, time %s' %
                  (no_saved, time.time()))
            model.save(no_saved)
            no_saved = no_saved + 1

        if time.time() >= train_start_time + (frequency_count * display_frequency):
            frequency_count = frequency_count + 1
            visualizer.display_current_results(model.get_current_visuals(), no_saved, True)
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(no_saved, frequency_count, errors, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(no_saved, float(frequency_count), opt, errors)

    if time.time() >= train_end_time:
        print("EXITING TRAINING")
        sys.exit()
        break
    model.update_learning_rate()
