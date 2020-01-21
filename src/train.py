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
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Desired graphics card config
session_conf = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=False,
      gpu_options=tf.GPUOptions(allow_growth=True))

# Parameters
# ==================================================

TUNE_HYPERPARAMS = True

np.random.seed(10)


# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 300)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("learn_rate", 1e-03, "Learn rate for Adam optimizer (default: 1e-03)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.5, "L2 regularization lambda (default: 0.5)")
tf.flags.DEFINE_boolean("l2_all_layers", False, "Apply L2 regularization on all layers (default: False)")
tf.flags.DEFINE_boolean("early_stop", True, "Apply early stop check (default: True)")
tf.flags.DEFINE_integer("early_stopping_step", 5, "How many dev steps without improvement till early stop (default: 5)")
tf.flags.DEFINE_float("max_norm", 1, "Max-norm regularization threshold (default: 1)")
tf.flags.DEFINE_boolean("max_norm_all_layers", False, "Apply max-norm regularization on all layers (default: False)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 4096, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 100)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Tuning parameters
tf.flags.DEFINE_boolean("tune_hyperparams", False, "Tune the hyper parameters with a random search (default: False)")
tf.flags.DEFINE_integer("tune_iterations", 10, "Number of tuning iterations (default: 10)")
tf.flags.DEFINE_float("learn_rate_min", 1e-05, "Min value for tuning learn rate (default: 1e-05")
tf.flags.DEFINE_float("learn_rate_max", 1e-01, "Max value for tuning learn rate (default: 1e-01")
tf.flags.DEFINE_float("dropout_keep_prob_min", 0.1, "Min value for tuning dropout keep probability (default: 0.1)")
tf.flags.DEFINE_float("dropout_keep_prob_max", 1.0, "Max value for tuning dropout keep probability (default: 1.0)")
tf.flags.DEFINE_float("l2_reg_lambda_min", 0.0, "Min value for tuning L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_float("l2_reg_lambda_max", 1.0, "Max value for tuning L2 regularization lambda (default: 1.0)")
tf.flags.DEFINE_float("max_norm_reg_min", 1.0, "Min value for tuning max-norm threshold (default: 1.0)")
tf.flags.DEFINE_float("max_norm_reg_max", 4.0, "Max value for tuning max-norm threshold (default: 4.0)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

def preprocess():
    print("loading data...")
    x = pickle.load(open("./pickle.p","rb"))
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

    return x_train, y_train, x_dev, y_dev, x_test, y_test, W, word_idx_map, vocab, max_l

def train(x_train, y_train, x_dev, y_dev, x_test, y_test, W, word_idx_map, vocab, max_l, learn_rate = FLAGS.learn_rate, keep_prob = FLAGS.dropout_keep_prob, l2_reg_lambda = FLAGS.l2_reg_lambda, max_norm = FLAGS.max_norm):
# Training
# ==================================================
    with tf.Graph().as_default():

        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length = max_l,
                num_classes = len(y_train[0]) ,
                vocab_size = len(vocab),
                word2vec_W = W,
                word_idx_map = word_idx_map,
                embedding_size = FLAGS.embedding_dim,
                batch_size = FLAGS.batch_size,
                filter_sizes = list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters = FLAGS.num_filters,
                l2_reg_lambda = l2_reg_lambda,
                l2_all_layers = FLAGS.l2_all_layers,
                max_norm_all_layers = FLAGS.max_norm_all_layers)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(learn_rate)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            if FLAGS.max_norm_all_layers and max_norm > 0.:
                for var in tf.trainable_variables():
                    if "W" in var.name:
                        clipped_weights = tf.clip_by_norm(var, clip_norm=max_norm, axes=1)
                        clip_weights = tf.assign(var, clipped_weights)

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

            sess.run(tf.global_variables_initializer())


    def train_step(x_batch, y_batch):
        """
        A single training step
        """
        feed_dict = {
          cnn.input_x: x_batch,
          cnn.input_y: y_batch,
          cnn.dropout_keep_prob: keep_prob
        }
        _, step, summaries, loss, accuracy = sess.run(
            [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
            feed_dict)
        if FLAGS.max_norm_all_layers and FLAGS.max_norm > 0.:
            clip_weights.eval(session=sess)
        time_str = datetime.datetime.now().isoformat()
        train_summary_writer.add_summary(summaries, step)
        return loss, accuracy


    def dev_step(x_batch, y_batch):
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
        dev_summary_writer.add_summary(summaries, step)
        return loss, conf_mat


    # Generate train batches
    train_batches = data_helpers.batch_iter(
        list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
    
    train_losses = []
    train_accuracies = []

    # Early stop parameters
    early_train_loss, early_train_acc, early_dev_loss, early_dev_acc, early_test_loss, early_test_acc = None, None, 1000, 0, None, None

    # Training loop. For each train batch...
    for batch in train_batches:
        x_batch, y_batch = zip(*batch)

        loss, acc = train_step(x_batch, y_batch)
        train_losses.append(loss)
        train_accuracies.append(acc)

        current_step = tf.train.global_step(sess, global_step)

        if current_step % FLAGS.evaluate_every == 0:
            print("\nEvaluation for step " + str(current_step) + ":")
            print("")

            train_loss = np.mean(np.asarray(train_losses))
            train_acc = np.mean(np.asarray(train_accuracies))
            print("Train loss {:g}, Train acc {:g}".format(train_loss, train_acc))

            # Evaluate on validation set
            dev_batches = data_helpers.batch_iter_dev(list(zip(x_dev, y_dev)), FLAGS.batch_size)
            dev_losses = []
            dev_conf_mat = np.zeros((2,2))
            for dev_batch in dev_batches:
                x_dev_batch = x_dev[dev_batch[0]:dev_batch[1]]
                y_dev_batch = y_dev[dev_batch[0]:dev_batch[1]]
                loss, conf_mat = dev_step(x_dev_batch, y_dev_batch)
                dev_losses.append(loss)
                dev_conf_mat += conf_mat

            dev_loss = np.mean(np.asarray(dev_losses))
            dev_acc = float(dev_conf_mat[0][0]+dev_conf_mat[1][1])/len(y_dev)
            print("Valid loss {:g}, Valid acc {:g}".format(dev_loss, dev_acc))

            # Evaluate on test set
            test_batches = data_helpers.batch_iter_dev(list(zip(x_test, y_test)), FLAGS.batch_size)
            test_losses = []
            test_conf_mat = np.zeros((2, 2))
            for test_batch in test_batches:
                x_test_batch = x_test[test_batch[0]:test_batch[1]]
                y_test_batch = y_test[test_batch[0]:test_batch[1]]
                loss, conf_mat = dev_step(x_test_batch, y_test_batch)
                test_losses.append(loss)
                test_conf_mat += conf_mat

            test_loss = np.mean(np.asarray(test_losses))
            test_acc = float(test_conf_mat[0][0] + test_conf_mat[1][1]) / len(y_test)
            print("Test loss {:g}, Test acc {:g}".format(test_loss, test_acc))
            print("Test - Confusion Matrix: ")
            print(test_conf_mat)

            # Early stop check
            if FLAGS.early_stop:
                if dev_loss < early_dev_loss:
                    stopping_step = 0
                    early_train_loss, early_train_acc, early_dev_loss, early_dev_acc, early_test_loss, early_test_acc = train_loss, train_acc, dev_loss, dev_acc, test_loss, test_acc
                    # This is probably really efficient because it saves models too often
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model with best dev loss checkpoint to {}\n".format(path))
                else:
                    stopping_step += 1
                if stopping_step >= FLAGS.early_stopping_step:
                    print("Early stopping is triggered at step {}. Step {} lead to the best results:".format(current_step, current_step - FLAGS.early_stopping_step))
                    print("Best train loss {:g}, Best train acc {:g}".format(early_train_loss, early_train_acc))
                    print("Best valid loss {:g}, Best valid acc {:g}".format(early_dev_loss, early_dev_acc))
                    print("Best test loss {:g}, Best test acc {:g}".format(early_test_loss, early_test_acc))
                    print("The best model is the last saved model.")
                    return early_train_loss, early_train_acc, early_dev_loss, early_dev_acc, early_test_loss, early_test_acc

        if not FLAGS.early_stop and current_step % FLAGS.checkpoint_every == 0:
            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print("Saved model checkpoint to {}\n".format(path))

    return train_loss, train_acc, dev_loss, dev_acc, test_loss, test_acc


def generate_random_hyperparams(lr_min, lr_max, kp_min, kp_max, l2_min, l2_max, norm_min, norm_max):
    # random search through log space
    random_learning_rate = np.random.uniform(lr_min, lr_max)
    random_keep_prob = np.random.uniform(kp_min, kp_max)
    random_l2_reg_lambda = np.random.uniform(l2_min, l2_max)
    random_max_norm = np.random.uniform(norm_min, norm_max)
    return random_learning_rate, random_keep_prob, random_l2_reg_lambda, random_max_norm


def tune(x_train, y_train, x_dev, y_dev, x_test, y_test, W, word_idx_map, vocab, max_l):
    def print_result(result):
        print("Loss: " + str(result["loss"]) + 
                ", Accuracy: " + str(result["accuracy"]) + 
                ", Learn rate: " + str(result["learn_rate"]) + 
                ", Keep prob: " + str(result["keep_prob"]) + 
                ", L2 reg lambda: " + str(result["l2_reg_lambda"]) +
                ", Max_norm: " + str(result["max_norm"]))

    tune_results =  []
    for i in range(FLAGS.tune_iterations):
        learn_rate, keep_prob, l2_reg_lambda, max_norm = generate_random_hyperparams(FLAGS.learn_rate_min, FLAGS.learn_rate_max, FLAGS.dropout_keep_prob_min, FLAGS.dropout_keep_prob_max, FLAGS.l2_reg_lambda_min, FLAGS.l2_reg_lambda_max, FLAGS.max_norm_reg_min, FLAGS.max_norm_reg_max)
        train_loss, train_acc, dev_loss, dev_acc, test_loss, test_acc = train(x_train, y_train, x_dev, y_dev, x_test, y_test, W, word_idx_map, vocab, max_l, learn_rate, keep_prob, l2_reg_lambda)
        tune_result =  {
          "loss": dev_loss,
          "accuracy": dev_acc,
          "learn_rate": learn_rate,
          "keep_prob": keep_prob,
          "l2_reg_lambda": l2_reg_lambda,
          "max_norm": max_norm
        }
        tune_results.append(tune_result)
        print("Tuning iteration: " + str(i))
        print_result(tune_result)
   
    best_result = None
    best_loss = 1000
    best_acc = 0.0
    for i, result in enumerate(tune_results):
        print("Result " + str(i) + ":")
        print_result(result)
        if result["accuracy"] > best_acc:
            best_acc = result["accuracy"]
            best_result = result

    print("Best result:")
    print_result(best_result)


def main(argv=None):
    x_train, y_train, x_dev, y_dev, x_test, y_test, W, word_idx_map, vocab, max_l = preprocess()

    if FLAGS.tune_hyperparams:
        tune(x_train, y_train, x_dev, y_dev, x_test, y_test, W, word_idx_map, vocab, max_l)
    else:
        train(x_train, y_train, x_dev, y_dev, x_test, y_test, W, word_idx_map, vocab, max_l)


if __name__ == '__main__':
    tf.app.run()
