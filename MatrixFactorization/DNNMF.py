import tensorflow as tf
import numpy as np
import pandas as pd

class DNNMF:
    """
    input_matrix: pandas.DataFrame (zero represents to no ranking)
    """
    def __init__(self, input_matrix, num_layers,
                 rank=2, lr=0.001, steps=1000):
        self.matrix = input_matrix
        self.mask = input_matrix.replace(0, np.NAN).notnull()
        self.num_layers = num_layers
        self.rank = rank
        self.shape = input_matrix.values.shape
        self.lr = lr
        self.steps = steps
        self.inner_size = (self.shape[0] + self.shape[1]) * (1 + self.rank) * 3

    def _layers(self, flatten_matrix):
        layer_list = []
        for i in xrange(self.num_layers):
            if i == 0:
                layer = tf.layers.dense(inputs=flatten_matrix,
                                        units=self.inner_size,
                                        activation=tf.nn.relu)
                layer_list.append(layer)
            else:
                layer = tf.layers.dense(inputs=layer_list[-1],
                                        units=self.inner_size,
                                        activation=tf.nn.relu)
        return layer_list[-1]

    def _model(self):
        tf_mask = tf.Variable(self.mask)
        tf_matrix = tf.constant(self.matrix.values)
        flatten_matrix = tf.reshape(tf_matrix, [1, -1])

        output_layer = self._layers(flatten_matrix)

        wsize = self.shape[0]*self.rank
        hsize = self.rank*self.shape[1]
        wbsize = self.shape[0]
        hbsize = self.shape[1]

        w = tf.maximum(tf.Variable(output_layer[0, :wsize]), 0)
        h = tf.maximum(tf.Variable(output_layer[0, wsize:wsize+hsize]), 0)
        W = tf.reshape(w, [self.shape[0], self.rank])
        H = tf.reshape(h, [self.rank, self.shape[1]])

        BW = tf.maximum(tf.Variable(output_layer[0, wsize+hsize:wsize+hsize+wbsize]), 0)
        BH = tf.maximum(tf.Variable(output_layer[0, -hbsize:]), 0)

        wh1 = tf.nn.bias_add(tf.matmul(W, H), BH)
        wh2 = tf.transpose(wh1, perm=[1, 0])
        wh3 = tf.nn.bias_add(wh2, BW)
        WH = tf.transpose(wh3, perm=[1, 0])

        # base_cost
        base_cost = tf.reduce_sum(tf.pow(tf.boolean_mask(tf_matrix, tf_mask) - tf.boolean_mask(WH, tf_mask), 2))

        # regularization
        l = tf.constant(0.1)
        matrix_sums = tf.add(tf.reduce_sum(tf.pow(H, 2)), tf.reduce_sum(tf.pow(W, 2)))
        bias_sums = tf.add(tf.reduce_sum(tf.pow(BW, 2)), tf.reduce_sum(tf.pow(BH, 2)))

        tmp = tf.clip_by_value(WH, 1.0, 5.0)
        penalty = tf.reduce_sum(tf.pow(tf.boolean_mask(WH, ~tf_mask) - tf.boolean_mask(tmp, ~tf_mask), 2))

        parameter_sums = tf.add(matrix_sums, bias_sums)
        parameter_sums = tf.add(parameter_sums, penalty)
        regularizer = tf.multiply(parameter_sums, l)

        # cost = base_cost + regularization
        cost = tf.add(base_cost, regularizer)

        return W, H, BW, BH, WH, base_cost

    def train(self):
        W, H, bw, bh, WH, cost = self._model()

        #train_step = tf.train.GradientDescentOptimizer(self.lr).minimize(cost)
        train_step = tf.train.RMSPropOptimizer(self.lr).minimize(cost)
        #train_step = tf.train.AdamOptimizer(self.lr).minimize(cost)

        init = tf.global_variables_initializer()

        prev_cost = 1e10
        count = 0
        with tf.Session() as sess:
            sess.run(init)
            for i in xrange(self.steps):
                sess.run(train_step)
                if not i%1000:
                    curr_cost = sess.run(cost)
                    print '\nRound: {0}, Cost: {1}'.format(i, curr_cost)
                    print '=' * 50
                    if curr_cost >= prev_cost:
                        count += 1
                    prev_cost = curr_cost
            learnt_W = sess.run(W)
            learnt_H = sess.run(H)
            learnt_bw = sess.run(bw)
            learnt_bh = sess.run(bh)
            learnt_WH = sess.run(WH)

        return learnt_W, learnt_H, learnt_bw, learnt_bh, learnt_WH
