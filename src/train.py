#!/usr/bin/env python
import pickle
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
import sys
from text_cnn import TextCNN
import os
from tensorflow.contrib import learn
import csv
from time import sleep
import pickle
import matplotlib.pyplot as plt



#####################  GPU Configs  #################################

# Selecting the GPU to work on
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# Desired graphics card config
session_conf = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=False,
      gpu_options=tf.GPUOptions(allow_growth=True))

# Parameters
# ==================================================

np.random.seed(10)


# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.5, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 4096, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

def preprocess():
    print("loading data...")
    x = pickle.load(open("./mainbalancedpickle.p","rb"))
    revs, W, W2, word_idx_map, vocab, max_l = x[0], x[1], x[2], x[3], x[4], x[5]
    print("data loaded!")# Load data

    max_l = 100

    x_text = []
    y = []

    test_x = []
    test_y = []

    for i in range(len(revs)):
        if revs[i]['split']==1:
            x_text.append(revs[i]['text'])
            temp_y = revs[i]['label']
            y.append(temp_y)
        else:
            test_x.append(revs[i]['text'])
            test_y.append(revs[i]['label'])

    y = np.asarray(y)
    test_y = np.asarray(test_y)

    print(x_text)
    print(test_x)

    # get word indices
    x = []
    for i in range(len(x_text)):
        x.append(np.asarray([word_idx_map[word] for word in x_text[i].split()]))

    x_test = []
    for i in range(len(test_x)):
        x_test.append(np.asarray([word_idx_map[word] for word in test_x[i].split()]))

    # padding
    for i in range(len(x)):
        if( len(x[i]) < max_l ):
            x[i] = np.append(x[i],np.zeros(max_l-len(x[i])))
        elif( len(x[i]) > max_l ):
            x[i] = x[i][0:max_l]
    x = np.asarray(x)

    for i in range(len(x_test)):
        if( len(x_test[i]) < max_l ):
            x_test[i] = np.append(x_test[i],np.zeros(max_l-len(x_test[i])))
        elif( len(x_test[i]) > max_l ):
            x_test[i] = x_test[i][0:max_l]
    x_test = np.asarray(x_test)
    y_test = test_y

    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    x_train = np.asarray(x_train)
    x_dev = np.asarray(x_dev)
    y_train = np.asarray(y_train)
    y_dev = np.asarray(y_dev)
    word_idx_map["@"] = 0
    rev_dict = {v: k for k, v in word_idx_map.items()}

    return x_train, y_train, x_dev, y_dev, W, word_idx_map, vocab, max_l

def train(x_train, y_train, x_dev, y_dev, W, word_idx_map, vocab, max_l):
# Training
# ==================================================
    with tf.Graph().as_default():

        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=max_l,
                num_classes=len(y_train[0]) ,
                vocab_size=len(vocab),
                word2vec_W = W,
                word_idx_map = word_idx_map,
                embedding_size=FLAGS.embedding_dim,
                batch_size=FLAGS.batch_size,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            #vocab_processor.save(os.path.join(out_dir, "vocab"))

            sess.run(tf.global_variables_initializer())


    def train_step(x_batch,
        # author_batch, topic_batch,
        y_batch):
        """
        A single training step
        """
        feed_dict = {
          cnn.input_x: x_batch,
          cnn.input_y: y_batch,
          cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
        }
        _, step, summaries, loss, accuracy = sess.run(
            [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
            feed_dict)
        time_str = datetime.datetime.now().isoformat()
        train_summary_writer.add_summary(summaries, step)
        return loss, accuracy

    def dev_step(x_batch, y_batch, writer=None):
        """
        Evaluates model on a dev set
        """
        feed_dict = {
          cnn.input_x: x_batch,
          cnn.input_y: y_batch,
          cnn.dropout_keep_prob: 1.0
        }
        step, summaries, loss, conf_mat = sess.run(
            [global_step, dev_summary_op, cnn.loss, cnn.confusion_matrix],
            feed_dict)
        if writer:
            writer.add_summary(summaries, step)
        return loss, conf_mat


    # Generate batches
    batches = data_helpers.batch_iter(
        list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
    # Training loop. For each batch...
    for batch in batches:
        x_batch, y_batch = zip(*batch)
        train_step(x_batch, y_batch)
        dev_step(x_dev, y_dev, writer=dev_summary_writer)
        current_step = tf.train.global_step(sess, global_step)
        if current_step % FLAGS.evaluate_every == 0:
            print("\nEvaluation:")
            #dev_step(x_dev, y_dev, writer=dev_summary_writer)
            print("")
        if current_step % FLAGS.checkpoint_every == 0:
            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print("Saved model checkpoint to {}\n".format(path))


#dev_batches = data_helpers.batch_iter(list(zip(x_dev, y_dev)), FLAGS.batch_size, FLAGS.num_epochs)
# Training loop. For each batch...
'''
train_loss = []
train_acc = []
best_acc = 0
for batch in batches:
    x_batch, y_batch = zip(*batch)
    x_batch = np.asarray(x_batch)
    y_batch = np.asarray(y_batch)

    t_loss, t_acc = train_step(x_batch, y_batch)
    current_step = tf.train.global_step(sess, global_step)
    train_loss.append(t_loss)
    train_acc.append(t_acc)
    print('Loss at step %s: %s' % (current_step, t_loss))
    print('Accuracy at step %s: %s' % (current_step, t_acc))

    if current_step % FLAGS.evaluate_every == 0:
        print(current_step)
        print("Train loss {:g}, Train acc {:g}".format(np.mean(np.asarray(train_loss)), np.mean(np.asarray(train_acc))))
        train_loss = []
        train_acc = []
        # Divide into batches
        dev_batches = data_helpers.batch_iter_dev(list(zip(x_dev, y_dev)), FLAGS.batch_size)
        dev_loss = []
        ll = len(dev_batches)
        conf_mat = np.zeros((2,2))
        for dev_batch in dev_batches:
            x_dev_batch = x_dev[dev_batch[0]:dev_batch[1]]
            y_dev_batch = y_dev[dev_batch[0]:dev_batch[1]]
            a, b = dev_step(x_dev_batch, y_dev_batch, writer=dev_summary_writer)
            dev_loss.append(a)
            conf_mat += b
        valid_accuracy = float(conf_mat[0][0]+conf_mat[1][1])/len(y_dev)
        print("Valid loss {:g}, Valid acc {:g}".format(np.mean(np.asarray(dev_loss)), valid_accuracy))
        print("Valid - Confusion Matrix: ")
        print(conf_mat)
        test_batches = data_helpers.batch_iter_dev(list(zip(x_test, y_test)), FLAGS.batch_size)
        test_loss = []
        conf_mat = np.zeros((2,2))
        for test_batch in test_batches:
            x_test_batch = x_test[test_batch[0]:test_batch[1]]
            y_test_batch = y_test[test_batch[0]:test_batch[1]]
            a, b = dev_step(x_test_batch, 
                y_test_batch, writer=dev_summary_writer)
            test_loss.append(a)
            conf_mat += b
        print("Test loss {:g}, Test acc {:g}".format(np.mean(np.asarray(test_loss)), float(conf_mat[0][0]+conf_mat[1][1])/len(y_test)))
        print("Test - Confusion Matrix: ")
        print(conf_mat)
        sys.stdout.flush()
        if best_acc < valid_accuracy:
            best_acc = valid_accuracy
            directory = "./models"
            if not os.path.exists(directory):
                os.makedirs(directory)
            saver.save(sess, directory+'/main_balanced_user_plus_topic', global_step=1)
'''

def main(argv=None):
    x_train, y_train, x_dev, y_dev, W, word_idx_map, vocab, max_l = preprocess()
    train(x_train, y_train, x_dev, y_dev, W, word_idx_map, vocab, max_l)

if __name__ == '__main__':
    tf.app.run()
