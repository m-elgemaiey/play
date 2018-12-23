from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#print(mnist.train)

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
loss_summary = tf.summary.scalar('Loss', cross_entropy)

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
acc_summary = tf.summary.scalar('Accuracy', accuracy)

train_writer = tf.summary.FileWriter('./log/LogReg/train')
test_writer = tf.summary.FileWriter('./log/LogReg/test')

for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  _, lossVal, lossSumm = sess.run([train_step,cross_entropy, loss_summary], feed_dict={x: batch_xs, y_: batch_ys})
  acc, acc_summ = sess.run([accuracy, acc_summary], feed_dict={x: mnist.test.images, y_: mnist.test.labels})
  print("i:", i, "Accuracy:", acc, "Loss:", lossVal)
  train_writer.add_summary(lossSumm, i)
  test_writer.add_summary(acc_summ, i)

sess.close()
