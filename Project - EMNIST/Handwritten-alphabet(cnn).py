import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import tensorflow as tf


train = pd.read_csv('emnist/emnist-balanced-train.csv', header=None)
#train = pd.read_csv('emnist/emnist-digits-train.csv', header=None)

test = pd.read_csv('emnist/emnist-balanced-test.csv', header=None)
#test = pd.read_csv('emnist/emnist-digits-test.csv', header=None)

#train.head()

# Number of Train
numTrain = len(train)
# Number of Test
numTest = len(test)
# Number of Classes
numClasses = len(train[0].unique())
learning_rate = 0.001

train_data = train.iloc[:, 1:]
train_labels = train.iloc[:, 0]
test_data = test.iloc[:, 1:]
test_labels = test.iloc[:, 0]

# one-hot encoding
train_labels = pd.get_dummies(train_labels)
test_labels = pd.get_dummies(test_labels)
#train_labels.head()

train_data = train_data.values
train_labels = train_labels.values
test_data = test_data.values
test_labels = test_labels.values


rand = random.randint(0, numTrain - 1)

'''
plt.imshow(
    train_data[rand].reshape([28, 28]), 
    cmap='Greys'
    )
plt.show()
'''

def rotate(image):
    image = image.reshape([28, 28])
    image = np.fliplr(image)
    image = np.rot90(image)
    return image.reshape([28 * 28])
train_data = np.apply_along_axis(rotate, 1, train_data)/255
test_data = np.apply_along_axis(rotate, 1, test_data)/255

'''
plt.imshow(
    train_data[rand].reshape([28, 28]), 
    cmap='Greys'
    )
plt.show()
'''

tf.set_random_seed(777)  # for reproducibility
tf.reset_default_graph()

# EMNIST data image of shape 28 * 28 = 784
X = tf.placeholder(tf.float32, [None, 784])
# 47 classes: 10 digits, 26 letters, and 11 capital letters 
Y = tf.placeholder(tf.float32, [None, numClasses])
X_img = tf.reshape(X, [-1, 28, 28, 1])

# dropout (keep_prob) rate  0.7 on training, but should be 1 for testing
keep_prob = tf.placeholder(tf.float32)

# L1 Img Input shape=(n, 28, 28, 1)
# Conv = (n, 28, 28, 32)
# Pool = (n, 14, 14, 32)
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

# L2 Img Input shape=(n, 14, 14, 32)
# Conv = (n, 14, 14, 64)
# Pool = (n, 7, 7, 64)
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

# L3 Img Input shape=(n, 7, 7, 64)
# Conv = (n, 7, 7, 128)
# Pool = (n, 4, 4, 128)
# Reshape = (n, 4 * 4 * 128) 
W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
L3_flat = tf.reshape(L3, [-1, 128 * 4 * 4])

# L4 Fully Connected (FC) 4x4x128 inputs -> 625 outputs
W4 = tf.get_variable("W4", shape=[128*4*4, 625], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([625]))
L4 = tf.nn.relu(tf.matmul(L3_flat, W4)+b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

# L5 Final Fully Connected (FC) 625 inputs -> numClasses outputs
W5 = tf.get_variable("W5", shape=[625, numClasses], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([numClasses]))
logits = tf.matmul(L4, W5) + b5


# cost & train
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)



# parameters
num_epochs = 15
batch_size = 100
print('numTrain: ', numTrain)
num_iterations = int( numTrain / batch_size)

with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
    # Training
    for epoch in range(num_epochs):
        avg_cost = 0

        for i in range(num_iterations):
            batch_xs, batch_ys = train_data[i * 100: (i + 1) * 100], train_labels[i * 100: (i + 1) * 100]
            _, cost_val = sess.run([train, cost], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.7})
            avg_cost += cost_val / num_iterations

        print("Epoch: {:04d}, Cost: {:.9f}".format(epoch + 1, avg_cost))

    print("Learning finished")

    # Test model
    is_correct = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))

    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

    # Test the model using test sets
    print("Accuracy: ", sess.run(accuracy, feed_dict={X: test_data, Y: test_labels, keep_prob: 1}))
    

    # Get Label and predict
    r = random.randint(0, numTest - 1)
    print("Label: ", sess.run(tf.argmax(test_labels[r : r + 1], 1)))
    print(
        "Prediction: ",
        sess.run(tf.argmax(logits, 1), feed_dict={X: test_data[r : r + 1], keep_prob: 1}),
    )

    plt.imshow(
        test_data[r : r + 1].reshape(28, 28),
        cmap="Greys",
        interpolation="nearest",
    )
    plt.show()