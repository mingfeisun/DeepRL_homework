# import expert policy
import pickle
from sklearn.model_selection import train_test_split

with open('expert_data/Hopper-v2.pkl', 'rb') as fin:
    data = pickle.load(fin)
    data_x = data["observations"] # 20000 * 11
    data_y = data["actions"].reshape(20000, 3) # 20000 * 1 * 3
    x_train, x_test, y_train, y_test = train_test_split(
        data_x, data_y, test_size=0.3, shuffle=False)

import tensorflow as tf

# Parameters
learning_rate = 0.1
num_steps = 200
batch_size = 128

# Network Parameters
n_hidden_1 = 64 # 1st layer number of neurons
n_hidden_2 = 64 # 2nd layer number of neurons
num_input = 11 
num_classes = 3

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# Create model
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
prediction = neural_net(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.losses.mean_squared_error(Y, prediction))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
regression_loss_mean = tf.reduce_mean(tf.losses.mean_squared_error(Y, prediction))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

train_size = len(x_train)

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps+1):
        batch_x = x_train[((step-1)*batch_size)%train_size : (step*batch_size)%train_size, :]
        batch_y = y_train[((step-1)*batch_size)%train_size : (step*batch_size)%train_size, :]

        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        loss, acc = sess.run([loss_op, regression_loss_mean], 
            feed_dict={X: batch_x, Y: batch_y})
        print("Step " + str(step) + ", Minibatch Loss= " + \
                "{:.4f}".format(loss) + ", Training loss= " + \
                "{:.3f}".format(acc))

    print("Optimization Finished!")

    print("Testing loss:", \
        sess.run(regression_loss_mean, 
            feed_dict={X: x_test, Y: y_test}))
