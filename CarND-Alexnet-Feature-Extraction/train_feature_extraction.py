import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
from sklearn.utils import shuffle
import csv
import numpy as np

# TODO: Load traffic signs data.
beta = .01
rate = .001
BATCH_SIZE = 256
with open('train.p', mode='rb') as trn:
    train = pickle.load(trn)
    x_train, y_train = train['features'], train['labels']

with open('signnames.csv', 'r') as sgn:
    reader = csv.reader(sgn)
    sign_names = list(reader)[1::]

n_train = x_train.shape[0]
nb_classes = len(sign_names)

# TODO: Split data into training and validation sets.

x_train, x_validation, y_train, y_validation = train_test_split(x_train,y_train,test_size=0.2, random_state=0)


# TODO: Define placeholders and resize operation.

x = tf.placeholder(tf.float32, (None, 32,32,3),name='x')
y = tf.placeholder(tf.int32, name='y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
# TODO: Resize the images so they can be fed into AlexNet.
# HINT: Use `tf.image.resize_images` to resize the images
resized = tf.image.resize_images(x, [227,227], method=0, align_corners=False)

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.

shape = (fc7.get_shape().as_list()[-1], nb_classes)  # use this shape for the weight matrix
fc8_w = tf.Variable(tf.truncated_normal(shape=shape,mean=0,stddev=0.1))
fc8_b = tf.Variable(tf.zeros(nb_classes))
logits = tf.matmul(fc7,fc8_w) + fc8_b
probs = tf.nn.softmax(logits)

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.

one_hot_y = tf.one_hot(y, nb_classes)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data, dropout_prob):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout_prob})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

# TODO: Train and evaluate the feature extraction model.
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy + beta * tf.nn.l2_loss(fc8_w))
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

# Training
EPOCHS = 7
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #saver.restore(sess, tf.train.latest_checkpoint('.'))
    num_examples = len(x_train)

    print("Training... Total Epoch: ", EPOCHS)
    print()

    for i in range(EPOCHS):
        crossEntropy = 0;
        x_train, y_train = shuffle(x_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = x_train[offset:end], y_train[offset:end]
            print('Now Training batch start: ',offset,' To ',end)
            _, loss_val = sess.run([training_operation, cross_entropy],
                                   feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
            crossEntropy += sum(loss_val);
        print('After batch')
        train_accuracy = evaluate(x_train, y_train, 1.0)
        validation_accuracy = evaluate(x_validation, y_validation, 1.0)
        print("EPOCH {} ...".format(i + 1))
        #print("train Accuracy = {:.3f}".format(train_accuracy), "test Accuracy = {:.3f}".format(test_accuracy))
        print("validation Accuracy = {:.3f}".format(validation_accuracy))
        print("Cross_entropy = {:.3f}".format(crossEntropy))
        if crossEntropy < 900:
            print('Max Accuracy reached')
            break

    saver.save(sess, '.\alexnet')
    print("Model saved")