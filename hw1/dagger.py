# import expert policy
import pickle
import copy
import os
import gym
import tf_util
import numpy
import tensorflow as tf

from sklearn.model_selection import train_test_split

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    parser.add_argument('--num_rollouts', type=int, default=10,
                        help='Number of expert roll outs')

    args = parser.parse_args()

    env = gym.make(args.envname + "-v2")
    max_steps = env.spec.timestep_limit

    observations = []
    actions = []

    import load_policy
    policy_name = "experts/%s-v2.pkl"%(args.envname)
    expertPolicyFn = load_policy.load_policy(policy_name)

    with tf.Session():
        tf_util.initialize()
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            steps = 0
            while not done:
                action = expertPolicyFn(obs[None,:])
                observations.append(obs)
                actions.append(action[0])
                obs, r, done, _ = env.step(action)
                steps += 1
                # env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break

    expert_data = {'observations': numpy.array(observations),
                    'actions': numpy.array(actions)}

    # Parameters
    learning_rate = 0.1
    num_steps = 500
    batch_size = 128

    # input size 
    _, train_dim = expert_data["observations"].shape
    _, action_dim = expert_data["actions"].shape

    # Network Parameters
    n_hidden_1 = 128 # 1st layer number of neurons
    n_hidden_2 = 128 # 2nd layer number of neurons
    num_input = train_dim
    num_classes = action_dim

    # tf Graph input
    X = tf.placeholder("float", [None, num_input])
    Y = tf.placeholder("float", [None, num_classes])

    # Store layers weight & bias
    weights_tf = {
        'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
    }
    biases_tf = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([num_classes]))
    }

    # Create model
    def neural_net(_x):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(_x, weights_tf['h1']), biases_tf['b1']))
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights_tf['h2']), biases_tf['b2']))
        out_layer = tf.matmul(layer_2, weights_tf['out']) + biases_tf['out']
        return out_layer

    prediction = neural_net(X)

    loss_op = tf.reduce_mean(tf.losses.mean_squared_error(Y, prediction))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    regression_loss_mean = tf.reduce_mean(tf.losses.mean_squared_error(Y, prediction))
    tf.summary.scalar('loss', regression_loss_mean)

    init = tf.global_variables_initializer()

    def getPolicy(_weights, _biases):
        obs_bo = tf.placeholder(tf.float32, [None, None])
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(obs_bo, _weights['h1']), _biases['b1']))
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, _weights['h2']), _biases['b2']))
        a_ba = tf.matmul(layer_2, _weights['out']) + _biases['out']

        return tf_util.function([obs_bo], a_ba)

    def trainPolicy(_expert_data):
        data_x = _expert_data['observations']
        data_y = _expert_data["actions"]

        x_train = copy.deepcopy(data_x)
        y_train = copy.deepcopy(data_y)

        train_size = len(x_train)

        # Start training
        with tf.Session() as sess:
            train_writer = tf.summary.FileWriter('./logs/train', sess.graph)

            # Run the initializer
            sess.run(init)

            for step in range(1, num_steps+1):
                batch_x = x_train[((step-1)*batch_size)%train_size : (step*batch_size)%train_size, :]
                batch_y = y_train[((step-1)*batch_size)%train_size : (step*batch_size)%train_size, :]

                merge = tf.summary.merge_all()

                # sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
                summary, loss, acc = sess.run([merge, loss_op, regression_loss_mean], 
                    feed_dict={X: batch_x, Y: batch_y})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                        "{:.4f}".format(loss) + ", Training loss= " + \
                        "{:.3f}".format(acc))

                train_writer.add_summary(summary, step)

            tmp_weights = {"h1": weights_tf['h1'].eval(), 
                            "h2": weights_tf['h2'].eval(), 
                            "out": weights_tf['out'].eval()}
            tmp_biases = {"b1": biases_tf['b1'].eval(), 
                            "b2": biases_tf['b2'].eval(), 
                            "out": biases_tf['out'].eval()}
            
        return copy.deepcopy(tmp_weights), copy.deepcopy(tmp_biases)

    def runPolicy(_policy_fn):
        env = gym.make(args.envname + "-v2")
        max_steps = env.spec.timestep_limit
        with tf.Session():
            tf_util.initialize()
            observations = []
            actions = []

            obs = env.reset()
            steps = 0
            while steps < args.num_rollouts*1000:
                action = _policy_fn(obs[None,:])
                obs, r, done, _ = env.step(action)

                observations.append(obs)
                steps += 1
                env.render()
            
            return copy.deepcopy(observations)

    while True:
        weights, biases = trainPolicy(expert_data)
        curr_policy_fn = getPolicy(weights, biases)

        with tf.Session():
            tf_util.initialize()
            new_x = numpy.array(runPolicy(curr_policy_fn))
            new_y = numpy.array(expertPolicyFn(new_x))
            expert_data["observations"] = copy.deepcopy(numpy.vstack((expert_data["observations"], new_x)))
            expert_data["actions"] = copy.deepcopy(numpy.vstack((expert_data["actions"], new_y)))

if __name__ == "__main__":
    main()