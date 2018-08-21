import tensorflow as tf
import sys
import os

# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

image_path = sys.argv[1]

# Read the image_data
image_data = tf.gfile.FastGFile(image_path, 'rb').read()

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
               in tf.gfile.GFile("/tmp/output_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("/tmp/output_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    from glob import glob

    folder = 'test2Mackenko'
    size = 3000
    pattern = "*.png"
    print("TESTING FOLDER: ", folder)
    dir = '/home/tarek/Downloads/patches/REALPATCHES/' + folder + '/normal/'
    images = []

    for _, _, _ in os.walk(dir):
        images.extend(glob(os.path.join(dir, pattern)))
    dir = '/home/tarek/Downloads/patches/REALPATCHES/' + folder + '/tumor/'

    for _, _, _ in os.walk(dir):
        images.extend(glob(os.path.join(dir, pattern)))

    # images.sort()
    import random

    random.shuffle(images)
    images = images[0:size]
    import numpy as np

    y_true = []
    y_pred = []
    probabilities = []
    i = 0
    for img in images:

        img_name = (img.split('/')[-1])

        # Read the image_data
        image_data = tf.gfile.FastGFile(img, 'rb').read()

        # print(image_data)
        print("-------")
        import sys
        import cv2
        import numpy as np

        from matplotlib import pyplot as plt

        plt.show()
        img = cv2.imread('/home/tarek/Downloads/patches/REALPATCHES/test2GAN/tumor/tumor1.png')

        print(img)
        # VG
        import cv2
        import numpy as np
        from matplotlib import pyplot as plt

        # img = cv2.imread(img)
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            histr = cv2.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(histr, color=col)
        # plt.xlim([0, 256])
        plt.savefig('test2GAN.png')
        print("----")

        # imshow()

        plt.hist(img.ravel(), 256, [0, 256]);
        plt.savefig('tumor1_validation.png')
        print("asdasa")
        sys.exit()
        # print(i, img_name)
        if img_name.startswith("normal"):
            y_true.append(0)
        else:
            y_true.append(1)

        predictions = sess.run(softmax_tensor, \
                               {'DecodeJpeg/contents:0': image_data})

        print("predddd ", predictions)
        probabilities.append(predictions[0][0])
        if predictions[0][0] > 0.8:
            y_pred.append([True])
        else:
            y_pred.append([False])

        i = i + 1
        # print(predictions)

        # a5ro normal [1]
        # Sort to show labels of first prediction in order of confidence
        # top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        top_k = predictions[0].argsort()

        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            print('%s (score = %.5f)' % (human_string, score))
        print("-------------------------------")

    from sklearn.metrics import confusion_matrix

    probabilities = np.array(probabilities)

    print("-----")
    # y_pred = probabilities
    print("y_pred  aaaa", y_pred[:-10])
    print("y_pred  aaaa", y_pred)
    print("probabilities  aaaa", probabilities)
    print("-----")
    y_true = np.array(y_true)
    print("y_true  aaaa", y_true)

    confusion_mtx = (confusion_matrix(y_true, y_pred))
    tn = confusion_mtx[0, 0]
    fp = confusion_mtx[0, 1]
    fn = confusion_mtx[1, 0]
    tp = confusion_mtx[1, 1]
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    print("TESTING FOLDER: ", folder)
    print("TESTING set size: ", size)
    print(confusion_mtx)
    print("Accuracy: %.2f" % accuracy)
    print("Recall: %.2f" % recall)
    print("Precision: %.2f" % precision)
    print("F1 Score: %.2f" % f1_score)
