import tensorflow as tf
from tensorflow.contrib import rnn
import time
import datetime

#import mnist dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)

#define constants
#unrolled through 28 time steps
time_steps = 28
#hidden LSTM units
num_units = 64
#rows of 28 pixels
n_input = 28
#mnist is meant to be classified in 10 classes(0-9).
n_classes = 10
#size of batch
batch_size = 100
epochs = 40
batches_per_epoch = mnist.train.num_examples // batch_size


#weights and biases of appropriate shape to accomplish above task
out_weights = tf.Variable(tf.random_normal([num_units,n_classes]))
out_bias = tf.Variable(tf.random_normal([n_classes]))

#defining placeholders
#input image placeholder
x = tf.placeholder("float",[None, time_steps, n_input])
#input label placeholder
y = tf.placeholder("float",[None, n_classes])

#processing the input tensor from [batch_size,n_steps,n_input] to "time_steps" number of [batch_size,n_input] tensors
input=tf.unstack(x, time_steps, 1)

#defining the network
lstm_layer=rnn.BasicLSTMCell(num_units,forget_bias=1)
outputs,_=rnn.static_rnn(lstm_layer,input,dtype="float32")

#converting last output of dimension [batch_size,num_units] to [batch_size,n_classes] by out_weight multiplication
prediction = tf.matmul(outputs[-1], out_weights) + out_bias

#loss_function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=y))
#optimization
opt = tf.train.AdamOptimizer().minimize(loss)

#model evaluation
correct_prediction = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

summary_writer = tf.summary.FileWriter('logs/tf_lstm_' + time.strftime('%Y%m%d_%H%M%S'))

#initialize variables
init = tf.global_variables_initializer()

def run_epoch(batches_per_epoch, batch_src):
    epoch_acc = 0
    epoch_loss = 0
    for _ in range(batches_per_epoch):
        batch_xs, batch_ys = batch_src.next_batch(batch_size)
        batch_xs = batch_xs.reshape((batch_size, time_steps, n_input))
        _, lossVal, accVal = sess.run([opt, loss, accuracy], feed_dict={x: batch_xs, y: batch_ys})
        epoch_loss += lossVal
        epoch_acc += accVal
    epoch_acc /= batches_per_epoch
    epoch_loss /= batches_per_epoch

    return epoch_acc, epoch_loss

with tf.Session() as sess:
    sess.run(init)

    total_start_time = datetime.datetime.now()

    for i in range(epochs):
        start_time = datetime.datetime.now()
        epoch_acc, epoch_loss = run_epoch(batches_per_epoch, mnist.train)
        end_time = datetime.datetime.now()
        epoch_time = end_time - start_time
        print("Epoch:", i+1,  "  Accuracy:", epoch_acc, "Time", epoch_time)
        acc_summ = tf.Summary(value=[tf.Summary.Value(tag='acc', simple_value=epoch_acc)])
        loss_summ = tf.Summary(value=[tf.Summary.Value(tag='loss', simple_value=epoch_loss)])
        summary_writer.add_summary(loss_summ, i)
        summary_writer.add_summary(acc_summ, i)

    #calculating test accuracy
    batches_per_epoch = mnist.test.num_examples // batch_size
    epoch_acc, _ = run_epoch(batches_per_epoch, mnist.test)
    print('Testing accuracy:', epoch_acc)

    total_end_time = datetime.datetime.now()
    print("Total Time", total_end_time - total_start_time)

