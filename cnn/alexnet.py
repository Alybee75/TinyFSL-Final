import os
import sys
import time

import tensorflow as tf
import numpy as np
import cv2

#TODO apply weight_decay=0.0005 and check if training_accuracy frequency is ok

tf.app.flags.DEFINE_integer('training_epoch', 1,
                            'number of training epochs.')
tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory.')
FLAGS = tf.app.flags.FLAGS
OUTPUT_DIR = '/output'


class AlexNet(object):

    def __init__(self, x, keep_prob, num_classes=1000):
        # Parse input arguments into class variables
        self.X = x
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob

        # Call the create function to build the computational graph of AlexNet
        self.create()

    def create(self):

        self.X = tf.reshape(self.X, shape=[-1, 227, 227, 3])

        # 1st Layer: Conv (w ReLu) -> Lrn -> Pool
        conv1 = conv(self.X, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        norm1 = lrn(conv1, 2, 1e-05, 0.75, name='norm1')
        pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID', name='pool1')

        # 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool with 2 groups
        conv2 = conv(pool1, 5, 5, 256, 1, 1, groups=2, name='conv2')
        norm2 = lrn(conv2, 2, 1e-05, 0.75, name='norm2')
        pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID', name='pool2')

        # 3rd Layer: Conv (w ReLu)
        conv3 = conv(pool2, 3, 3, 384, 1, 1, name='conv3')

        # 4th Layer: Conv (w ReLu) splitted into two groups
        conv4 = conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')

        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        flattened = tf.reshape(pool5, [-1, 6 * 6 * 256])
        fc6 = fc(flattened, 6 * 6 * 256, 4096, name='fc6')
        dropout6 = dropout(fc6, self.KEEP_PROB)

        # 7th Layer: FC (w ReLu) -> Dropout
        fc7 = fc(dropout6, 4096, 4096, name='fc7')
        dropout7 = dropout(fc7, self.KEEP_PROB)

        # 8th Layer: FC and return unscaled activations
        self.fc8 = fc(dropout7, 4096, self.NUM_CLASSES, relu=False, name='fc8')
	
def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME', groups=1):
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)

    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable('weights', shape=[filter_height,
                                                    filter_width,
                                                    input_channels / groups,
                                                    num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])

    if groups == 1:
        conv = convolve(x, weights)

    # In the cases of multiple groups, split inputs & weights and
    else:
        # Split input and weights and convolve them separately
        input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
        weight_groups = tf.split(axis=3, num_or_size_splits=groups,
                                 value=weights)
        output_groups = [convolve(i, k) for i, k in
                         zip(input_groups, weight_groups)]

        # Concat the convolved output together again
        conv = tf.concat(axis=3, values=output_groups)

    # Add biases
    bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

    # Apply relu function
    relu = tf.nn.relu(bias, name=scope.name)

    return relu


def fc(x, num_in, num_out, name, relu=True):
    with tf.variable_scope(name) as scope:

        # Create tf variables for the weights and biases
        weights = tf.get_variable('weights', shape=[num_in, num_out],
                                  trainable=True)
        biases = tf.get_variable('biases', [num_out], trainable=True)

        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

    if relu:
        # Apply ReLu non linearity
        relu = tf.nn.relu(act)
        return relu
    else:
        return act


def max_pool(x, filter_height, filter_width, stride_y, stride_x, name,
             padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)


def lrn(x, radius, alpha, beta, name, bias=1.0):
    return tf.nn.local_response_normalization(x, depth_radius=radius,
                                              alpha=alpha, beta=beta,
                                              bias=bias, name=name)


def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)

class Dataset:
    ''' Class for handling Imagenet data '''

    def __init__(self, image_path):
        self.data = create_image_list(image_path)
        np.random.shuffle(self.data)
        self.num_records = len(self.data)
        self.next_record = 0

        self.labels, self.inputs = zip(*self.data)

        category = np.unique(self.labels)
        self.num_labels = len(category)
        self.category2label = dict(zip(category, range(len(category))))
        self.label2category = {l: k for k, l in self.category2label.items()}

        # Convert the labels to numbers
        self.labels = [self.category2label[l] for l in self.labels]

    def __len__(self):
        return self.num_records

    def onehot(self, label):
        v = np.zeros(self.num_labels)
        v[label] = 1
        return v

    def records_remaining(self):
        return len(self) - self.next_record

    def has_next_record(self):
        return self.next_record < self.num_records

    def preprocess(self, img):
        pp = cv2.resize(img, (227, 227))
        pp = np.asarray(pp, dtype=np.float32)
        pp /= 255
        pp = pp.reshape((pp.shape[0], pp.shape[1], 3))
        return pp

    def next_record_f(self):
        if not self.has_next_record():
            np.random.shuffle(self.data)
            self.next_record = 0
            self.labels, self.inputs = zip(*self.data)

            category = np.unique(self.labels)
            self.num_labels = len(category)
            self.category2label = dict(zip(category, range(len(category))))
            self.label2category = {l: k for k, l in self.category2label.items()}

            # Convert the labels to numbers
            self.labels = [self.category2label[l] for l in self.labels]
        # return None
        label = self.onehot(self.labels[self.next_record])
        input = self.preprocess(cv2.imread(self.inputs[self.next_record]))
        self.next_record += 1
        return label, input

    def next_batch(self, batch_size):
        records = []
        for i in range(batch_size):
            record = self.next_record_f()
            if record is None:
                break
            records.append(record)
        labels, input = zip(*records)
        return labels, input


def create_image_list(image_path):
    image_filenames = []
    category_list = [c for c in sorted(os.listdir(image_path))
                     if c[0] != '.' and
                     os.path.isdir(os.path.join(image_path, c))]
    for category in category_list:
        if category:
            walk_path = os.path.join(image_path, category)
        else:
            walk_path = image_path
            category = os.path.split(image_path)[1]

        w = _walk(walk_path)
        while True:
            try:
                dirpath, dirnames, filenames = w.next()
            except StopIteration:
                break
            # Don't enter directories that begin with '.'
            for d in dirnames[:]:
                if d.startswith('.'):
                    dirnames.remove(d)
            dirnames.sort()
            # Ignore files that begin with '.'
            filenames = [f for f in filenames if not f.startswith('.')]
            filenames.sort()

            for f in filenames:
                image_filenames.append([category, os.path.join(dirpath, f)])

    return image_filenames


def _walk(top):
    ''' Improved os.walk '''
    names = os.listdir(top)
    dirs, nondirs = [], []
    for name in names:
        if os.path.isdir(os.path.join(top, name)):
            dirs.append(name)
        else:
            nondirs.append(name)

    yield top, dirs, nondirs
    for name in dirs:
        path = os.path.join(top, name)
        for x in _walk(path):
            yield x

def main(_):
    # here we train and validate the model

    print 'Loading data'
    training = Dataset('/data/i1k-extracted/train')
    testing = Dataset('/data/i1k-extracted/val')
    print 'Data loaded.'

    if len(sys.argv) < 2 or sys.argv[-1].startswith('-'):
        print('Usage: alexnet.py [--training_epoch=x] '
              '[--model_version=y] export_dir')
        sys.exit(-1)
    if FLAGS.training_epoch <= 0:
        print
        'Please specify a positive value for training iteration.'
        sys.exit(-1)
    if FLAGS.model_version <= 0:
        print
        'Please specify a positive value for version number.'
        sys.exit(-1)

    batch_size = 128
    display_step = 20
    # training_acc_step = 1000 # think how to use it
    train_size = len(training)
    print train_size
    n_classes = training.num_labels
    print n_classes
    image_size = 227
    img_channel = 3
    num_epochs = FLAGS.training_epoch

    x_flat = tf.placeholder(tf.float32,
                            (None, image_size * image_size * img_channel))
    x_3d = tf.reshape(x_flat, shape=(tf.shape(x_flat)[0], image_size,
                                     image_size, img_channel))
    y = tf.placeholder(tf.float32, [None, n_classes])
    keep_prob = tf.placeholder(tf.float32)

    model = AlexNet(x_3d, keep_prob=keep_prob, num_classes=n_classes)
    model_train = model.fc8

    model_prediction = tf.nn.softmax(model_train)

    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=model_train, labels=y))
    global_step = tf.Variable(0, trainable=False, name='global_step')

    lr = tf.train.exponential_decay(0.01, global_step, 100000, 0.1,
                                    staircase=True)

    optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(cost, global_step=global_step)

    accuracy, update_op = tf.metrics.accuracy(labels=tf.argmax(y, 1),
                                              predictions=tf.argmax(model_prediction,
                                                                    1))
    test_accuracy, test_update_op = tf.metrics.accuracy(labels=tf.argmax(y, 1),
                                                        predictions=tf.argmax(
                                                            model_prediction, 1))

    start_time = time.time()
    print "Start time is: " + str(start_time)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        for step in xrange(int(num_epochs * train_size) // batch_size):

            batch_ys, batch_xs = training.next_batch(batch_size)

            sess.run(optimizer, feed_dict={x_3d: batch_xs, y: batch_ys, keep_prob: 0.5})
            sess.run(lr)
            if step % display_step == 0:
                acc_up = sess.run([accuracy, update_op],
                                  feed_dict={x_3d: batch_xs, y: batch_ys, keep_prob: 1.})
                acc = sess.run(accuracy,
                               feed_dict={x_3d: batch_xs, y: batch_ys, keep_prob: 1.})
                loss = sess.run(cost, feed_dict={x_3d: batch_xs, y: batch_ys, keep_prob: 1.})
                elapsed_time = time.time() - start_time
                print " Iter " + str(step) + ", Minibatch Loss= " + "{:.6f}".format(loss) + \
                ", Training Accuracy= " + "{}".format(acc) + " Elapsed time:" + str(elapsed_time) + \
                "acc_up={}".format(acc_up)

        print "Optimization Finished!"
        print "Training took" + str(time.time() - start_time)

        step_test = 1
        acc_list = []
        while step_test * batch_size < len(testing):
            testing_ys, testing_xs = testing.next_batch(batch_size)
            acc_up = sess.run([test_accuracy, test_update_op],
                              feed_dict={x_3d: testing_xs, y: testing_ys, keep_prob: 1.})
            acc = sess.run([test_accuracy],
                           feed_dict={x_3d: testing_xs, y: testing_ys, keep_prob: 1.})
            acc_list.append(acc)
            print "Testing Accuracy:", acc
            step_test += 1

        print "Max accuracy is", max(acc_list)
        print "Min accuracy is", min(acc_list)

        # save model using SavedModelBuilder from TF
        export_path_base = sys.argv[-1]
        export_path = os.path.join(
            tf.compat.as_bytes(export_path_base),
            tf.compat.as_bytes(str(FLAGS.model_version)))
        print 'Exporting trained model to', export_path
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)

        tensor_info_x = tf.saved_model.utils.build_tensor_info(x_flat)
        tensor_info_y = tf.saved_model.utils.build_tensor_info(model_train)

        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'images': tensor_info_x},
                outputs={'scores': tensor_info_y},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

        legacy_init_op = tf.group(tf.tables_initializer(),
                                  name='legacy_init_op')
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'predict_images':
                    prediction_signature,
            },
            legacy_init_op=legacy_init_op)

        builder.save()

        print 'Done exporting!'


if __name__ == '__main__':
    tf.app.run()