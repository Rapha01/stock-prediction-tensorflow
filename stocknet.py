# Import
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from pprint import pprint
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import data
data = pd.read_csv('stocks.csv')

# Dimensions of dataset
n = data.shape[0]
p = data.shape[1]

# Make data a np.array
data = data.values

# Training and test data
train_start = 0
train_end = int(np.floor(0.8*n))
test_start = train_end + 1
test_end = n
data_train = data[np.arange(train_start, train_end), :]
data_test = data[np.arange(test_start, test_end), :]

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
#scaler.fit(data_train)
#data_train = scaler.transform(data_train)
#data_test = scaler.transform(data_test)

scaler.fit(data_train[:, :-2])
# Build X and y
X_train = scaler.transform(data_train[:, :-2])
y_train = data_train[:, -1]
X_test = scaler.transform(data_test[:, :-2])
y_test = data_test[:, -1]

pprint(X_train);
pprint(y_train);

# Number of stocks in training data
n_inputs = X_train.shape[1]

# Neurons
n_neurons_1 = 100
n_neurons_2 = 100
n_neurons_3 = 100

# Session
net = tf.InteractiveSession()

# Placeholder
X = tf.placeholder(dtype=tf.float32, shape=[None, n_inputs])
Y = tf.placeholder(dtype=tf.float32, shape=[None])

# Initializers
sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.zeros_initializer()

# Hidden weights
W_hidden_1 = tf.Variable(weight_initializer([n_inputs, n_neurons_1]))
bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))


# Output weights
W_out = tf.Variable(weight_initializer([n_neurons_3, 1]))
bias_out = tf.Variable(bias_initializer([1]))

# Hidden layer
hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))


# Output layer (transpose!)
out = tf.transpose(tf.add(tf.matmul(hidden_3, W_out), bias_out))

# Cost function
mse = tf.reduce_mean(tf.squared_difference(out, Y))

# Optimizer
opt = tf.train.AdamOptimizer().minimize(mse)

# Init
net.run(tf.global_variables_initializer())

# Setup plot
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(111)
line1, = ax1.plot(y_test)
line2, = ax1.plot(y_test * 0.5)
plt.show()

def compute_forecast_accuracy(pred, y_test):
    p = []

    for i in range(0, len(pred[0]) - 1):
        if pred[0][i] > 0.65:
            if y_test[i] > 0.75:
                p.append(1)
            else:
                p.append(0)

        if pred[0][i] < 0.35:
            if y_test[i] < 0.25:
                p.append(1)
            else:
                p.append(0)
    accurate = 0

    for b in p:
        if b:
            accurate = accurate + 1

    if len(p) == 0:
        return 0.0

    return float(accurate) / float(len(p))



# Run
mse_train = []
mse_test = []

epochs = 1000000
for e in range(epochs):

    # Shuffle training data
    shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    X_train = X_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

    # Minibatch training

    batch_x = X_train
    batch_y = y_train
    # Run optimizer with batch
    net.run(opt, feed_dict={X: batch_x, Y: batch_y})

    # Show progress
    # MSE train and test
    mse_train.append(net.run(mse, feed_dict={X: X_train, Y: y_train}))
    mse_test.append(net.run(mse, feed_dict={X: X_test, Y: y_test}))
    print('MSE Train: ', mse_train[-1])
    print('MSE Test: ', mse_test[-1])
    # Prediction
    #print('Test Input')
    #pprint(X_test)
    #print('Test Output')
    #pprint(y_test)
    pred = net.run(out, feed_dict={X: X_test})
    print('Test Prediction')
    pprint(pred)

    line2.set_ydata(pred)
    plt.title('Epoch: ' + str(e) + ' Hitrate: ' + '%.2f' % (compute_forecast_accuracy(pred, y_test) * 100) + "%" + " MSE Train: " + '%.5f' % mse_train[-1] + " MSE Test: " + '%.5f' % mse_test[-1])
    print("Forecast hitrate: " + '%.2f' % (compute_forecast_accuracy(pred, y_test) * 100) + "%")
    plt.pause(0.0000001)
