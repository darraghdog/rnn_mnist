# Working example for my blog post at:
# http://danijar.com/variable-sequence-lengths-in-tensorflow/
'''
to activate py35 run 
    conda create -n py35 python=3.5 anaconda
    source activate py35
    # source deactivate py35
pip install tensorflow
pip install sets
'''

import functools
import sets
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
rnn_cell = tf.contrib.rnn
rnn = tf.nn


def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper


class VariableSequenceClassification:

    def __init__(self, data, target, num_hidden=200, num_layers=2):
        self.data = data
        self.target = target
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self.predict
        self.optimize

    @lazy_property
    def length(self):
        used = tf.sign(tf.reduce_max(tf.abs(self.data), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    @lazy_property
    def predict(self):
        # Recurrent network.
        length_ = self.length
        output, _ = rnn.dynamic_rnn(
            rnn_cell.GRUCell(self._num_hidden),
            data,
            dtype=tf.float32,
            sequence_length=self.length,
        )
        last = self._last_relevant(output, self.length)
        in_size = self._num_hidden
        out_size = int(self.target.get_shape()[1])
        weight = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.01))
        bias = tf.Variable(tf.constant(0.1, shape=[out_size]))
        # Softmax layer.
        prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
        return prediction, length_

    @lazy_property
    def optimize(self):
        learning_rate = 0.003
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
        cost = -tf.reduce_sum(self.target * tf.log(self.predict[0]))
        return optimizer.minimize(cost)

    @staticmethod
    def _last_relevant(output, length):
        batch_size = tf.shape(output)[0]
        max_length = int(output.get_shape()[1])
        output_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, output_size])
        relevant = tf.gather(flat, index)
        return relevant


if __name__ == '__main__':
    # We treat images as sequences of pixel rows.
    train, test = sets.Mnist()
    y_act  = test.target
    _, rows, row_size = train.data.shape
    num_classes = train.target.shape[1]
    data = tf.placeholder(tf.float32, [None, rows, row_size])
    target = tf.placeholder(tf.float32, [None, num_classes])
    model = VariableSequenceClassification(data, target)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    for epoch in range(3):
        for _ in range(100):
            batch = train.sample(10)
            sess.run(model.optimize, {data: batch.data, target: batch.target})
        #y_pred, y_last = sess.run(model.prediction, {data: test.data, target: test.target})
        y_pred, length = sess.run(model.predict, {data: test.data, target: test.target})
        print(length.tolist()[-5:])
        error  = np.mean(np.argmax(y_act, 1) != np.argmax(y_pred, 1))
        print(50*'=')
        print('Epoch {:2d} error {:3.1f}%'.format(epoch + 1, 100 * error))
        print(confusion_matrix(np.argmax(y_act, 1), np.argmax(y_pred, 1)))

        print(50*'=')
        

'''
    def length(self):
        used = tf.sign(tf.reduce_max(tf.abs(self.data), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length
'''
data = test.data[-5:]
datalen = length[-5:]
used = np.sign(np.max(np.abs(data), 2))
np.sum(used, 1)


