#!/usr/bin/env python
import tensorflow as tf
import numpy as np


def log_dir(rate, optimizer_class):
    return './logs/' + '_rate=' + str(rate) + '_optimizer=' + str(optimizer_class.__name__)


# Parameters
learning_rates = np.arange(0.001, 3.001, 0.001)  # Greater than 0
training_iteration = 100
optimizers = [tf.train.GradientDescentOptimizer, tf.train.AdamOptimizer, tf.train.AdagradOptimizer]


for learning_rate in learning_rates:
    for optimizer in optimizers:
        # Reset graph
        tf.reset_default_graph()

        # Model data holders
        x = tf.placeholder(tf.float32)
        y = tf.placeholder(tf.float32)

        # Model variable parameters
        W = tf.Variable([.3], dtype=tf.float32)
        b = tf.Variable([-.3], dtype=tf.float32)

        # Linear model definition
        model = W * x + b

        # Loss / cost function
        loss_function = tf.reduce_sum(tf.square(model - y))  # sum of the squares

        # training data
        x_train = [1, 2, 3, 4]
        y_train = [0, -1, -2, -3]

        # Tensorboard summaries
        tf.summary.scalar("loss_function", loss_function)
        tf.summary.scalar("learning_rate", learning_rate)
        tf.summary.histogram("weights", W)
        tf.summary.histogram("biases", b)

        # Gradient descent
        train = optimizer(learning_rate=learning_rate).minimize(loss_function)

        # training loop
        init = tf.global_variables_initializer()

        # Launch the graph
        sess = tf.Session()
        sess.run(init)

        # Write summaries
        summary_writer = tf.summary.FileWriter(log_dir(learning_rate, optimizer), sess.graph)
        merged_summary = tf.summary.merge_all()

        for i in range(training_iteration):
            curr_W, curr_b, curr_loss = sess.run([W, b, loss_function], {x: x_train, y: y_train})
            summary = sess.run(merged_summary, {x: x_train, y: y_train})
            summary_writer.add_summary(summary, i)
            sess.run(train, {x: x_train, y: y_train})
            print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss))

        # Flush writer
        summary_writer.flush()





