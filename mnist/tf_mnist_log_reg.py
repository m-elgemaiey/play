# Logistic Regression
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import time
import numpy as np

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#print(mnist.train)

# define the model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
#train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

summary_writer = tf.summary.FileWriter('logs/tf_logreg_' + time.strftime('%Y%m%d_%H%M%S'))

epochs = 50
batch_size = 100
batches_per_epoch = 60000 // batch_size
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for i in range(epochs):
    accuracies = []
    losses = []
    for _ in range(batches_per_epoch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, loss, acc = sess.run([train_step, cross_entropy, accuracy], feed_dict={x: batch_xs, y_: batch_ys})
        losses.append(loss)
        accuracies.append(acc)
    epoch_acc = np.mean(accuracies)
    epoch_loss = np.mean(losses)
    print("Epoch:", i+1,  "  Accuracy:", epoch_acc)
    acc_summ = tf.Summary(value=[tf.Summary.Value(tag='acc', simple_value=epoch_acc)])
    loss_summ = tf.Summary(value=[tf.Summary.Value(tag='loss', simple_value=epoch_loss)])
    summary_writer.add_summary(loss_summ, i)
    summary_writer.add_summary(acc_summ, i)

print('Testing accuracy:', sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
sess.close()
