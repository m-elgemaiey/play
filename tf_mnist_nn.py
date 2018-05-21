# Tensorflow NN 
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import time

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#print(mnist.train)

input = tf.placeholder(tf.float32, [None, 784])
labels = tf.placeholder(tf.float32, [None, 10])

hidden = tf.layers.dense(inputs=input, units=512, activation=tf.nn.relu)
hidden = tf.layers.dense(inputs=hidden, units=512, activation=tf.nn.relu)
#hidden = tf.layers.dense(inputs=hidden, units=1024, activation=tf.nn.relu)
#hidden = tf.layers.dense(inputs=hidden, units=1024, activation=tf.nn.relu)
output = tf.layers.dense(inputs=hidden, units=10)

loss = tf.losses.softmax_cross_entropy(labels, output)
train_step = tf.train.AdamOptimizer().minimize(loss)
correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

summary_writer = tf.summary.FileWriter('logs/tf_nn_' + time.strftime('%Y%m%d_%H%M%S'))

epochs = 60
batch_size = 100
batches_per_epoch = 60000 // batch_size

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for i in range(epochs):
    accuracies = []
    losses = []
    for _ in range(batches_per_epoch):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        _, lossVal, accVal = sess.run([train_step, loss, accuracy], feed_dict={input: batch_xs, labels: batch_ys})
        losses.append(lossVal)
        accuracies.append(accVal)
    epoch_acc = np.mean(accuracies)
    epoch_loss = np.mean(losses)
    print("Epoch:", i+1,  "  Accuracy:", epoch_acc)
    acc_summ = tf.Summary(value=[tf.Summary.Value(tag='acc', simple_value=epoch_acc)])
    loss_summ = tf.Summary(value=[tf.Summary.Value(tag='loss', simple_value=epoch_loss)])
    summary_writer.add_summary(loss_summ, i)
    summary_writer.add_summary(acc_summ, i)

print('Testing accuracy:', sess.run(accuracy, feed_dict={input: mnist.test.images, labels: mnist.test.labels}))
sess.close()
