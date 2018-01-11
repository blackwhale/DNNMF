import tensorflow as tf
import numpy as np
import pandas as pd

class MatrixFactorization:
    """
    input_matrix: pandas.DataFrame (zero represents to no ranking)
    """
    def __init__(self, input_matrix, rank=2, lr=0.001, steps=1000):
        self.matrix = input_matrix
        self.mask = input_matrix.replace(0, np.NAN).notnull()
        self.rank = rank
        self.shape = input_matrix.values.shape
        self.lr = lr
        self.steps = steps
        self.inner_size = self.shape[0] * self.shape[1] * self.rank

    def _model(self, enable_b=False):
        tf_mask = tf.Variable(self.mask)
        tf_matrix = tf.constant(self.matrix.values)
        flatten_matrix = tf.reshape(tf_matrix, [1, -1])

        l1 = tf.layers.dense(inputs=flatten_matrix,
                             units=self.inner_size,
                             activation=tf.nn.relu)
        l2 = tf.layers.dense(inputs=l1,
                             units=self.inner_size,
                             activation=tf.nn.relu)
        l3 = tf.layers.dense(inputs=l2,
                             units=self.inner_size,
                             activation=tf.nn.relu)
        l4 = tf.layers.dense(inputs=l3,
                             units=self.shape[0]*self.rank+self.shape[1]*self.rank+self.shape[0]*self.shape[1],
                             activation=tf.nn.relu)

        w = tf.maximum(tf.Variable(l4[0, :self.shape[0]*self.rank]), 0)
        h = tf.maximum(tf.Variable(l4[0, self.shape[0]*self.rank:-1*self.shape[0]*self.shape[1]]), 0)
        b = tf.maximum(tf.Variable(l4[0, -1*self.shape[0]*self.shape[1]:]), 0)

        W = tf.reshape(w, [self.shape[0], self.rank])
        H = tf.reshape(h, [self.rank, self.shape[1]])
        B = tf.reshape(b, [self.shape[0], self.shape[1]])

        WH = tf.add(tf.matmul(W, H), B)

        # base_cost
        base_cost = tf.reduce_sum(tf.pow(tf.boolean_mask(tf_matrix, tf_mask) - tf.boolean_mask(WH, tf_mask), 2))

        # regularization
        l = tf.constant(0.01)
        parameter_sums = tf.add(tf.reduce_sum(tf.pow(H, 2)), tf.reduce_sum(tf.pow(W, 2)))
        regularizer = tf.multiply(parameter_sums, l)

        # cost = base_cost + regularization
        cost = tf.add(base_cost, regularizer)

        return W, H, B, base_cost

    def train(self, enable_b=True):
        W, H, b, cost = self._model(enable_b)

        #train_step = tf.train.GradientDescentOptimizer(self.lr).minimize(cost)
        train_step = tf.train.RMSPropOptimizer(self.lr).minimize(cost)
        #train_step = tf.train.AdamOptimizer().minimize(cost)
        init = tf.global_variables_initializer()

        prev_cost = 1e10
        count = 0
        with tf.Session() as sess:
            sess.run(init)
            for i in xrange(self.steps):
                sess.run(train_step)
                if not i%1000:
                    curr_cost = sess.run(cost)
                    print '\nCost: {0}'.format(curr_cost)
                    print '=' * 50
                    if curr_cost >= prev_cost:
                        count += 1
                    #if count >= 10:
                    #    break
                    prev_cost = curr_cost
            learnt_W = sess.run(W)
            learnt_H = sess.run(H)
            learnt_b = sess.run(b)

        return learnt_W, learnt_H, learnt_b
