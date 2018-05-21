from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import time

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#print(mnist.train)

input = tf.placeholder(tf.float32, [None, 784])
labels = tf.placeholder(tf.float32, [None, 10])

input2d = tf.reshape(input, [-1,28,28,1])
conv1 = tf.layers.conv2d(inputs=input2d, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
pool_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

hidden = tf.layers.dense(inputs=pool_flat, units=1024, activation=tf.nn.relu)
should_drop = tf.placeholder(tf.bool)
dropout = tf.layers.dropout(inputs=hidden, rate=0.5, training=should_drop)
output = tf.layers.dense(inputs=dropout, units=10)

loss = tf.losses.softmax_cross_entropy(labels, output)
loss_summary = tf.summary.scalar('loss', loss)
train_step = tf.train.AdamOptimizer().minimize(loss)
correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

epochs = 30
batch_size = 100
batches_per_epoch = mnist.train.num_examples // batch_size

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

summary_writer = tf.summary.FileWriter('logs/tf_cnn_' + time.strftime('%Y%m%d_%H%M%S'))

def run_epoch(batches_per_epoch, batch_src, is_train):
    epoch_acc = 0
    epoch_loss = 0
    for _ in range(batches_per_epoch):
        batch_xs, batch_ys = batch_src.next_batch(100)
        _, lossVal, accVal = sess.run([train_step, loss, accuracy], feed_dict={input: batch_xs, labels: batch_ys, should_drop: is_train})
        epoch_loss += lossVal
        epoch_acc += accVal
    epoch_acc /= batches_per_epoch
    epoch_loss /= batches_per_epoch

    return epoch_acc, epoch_loss

for i in range(epochs):
    epoch_acc, epoch_loss = run_epoch(batches_per_epoch, mnist.train, True)
    print("Epoch:", i+1,  "  Accuracy:", epoch_acc)
    acc_summ = tf.Summary(value=[tf.Summary.Value(tag='acc', simple_value=epoch_acc)])
    loss_summ = tf.Summary(value=[tf.Summary.Value(tag='loss', simple_value=epoch_loss)])
    summary_writer.add_summary(loss_summ, i)
    summary_writer.add_summary(acc_summ, i)

batches_per_epoch = mnist.test.num_examples // batch_size
epoch_acc, _ = run_epoch(batches_per_epoch, mnist.test, False)
print('Testing accuracy:', epoch_acc)
sess.close()
