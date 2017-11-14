# Working example for my blog post at:
# http://danijar.com/variable-sequence-lengths-in-tensorflow/
import functools
import sets
import numpy as np
import tensorflow as tf
# from tensorflow.models.rnn import rnn_cell
# from tensorflow.models.rnn import rnn
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


class VariableSequenceLabelling:

    def __init__(self, data, target, num_hidden=200, num_layers=3):
        self.data = data
        self.target = target
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self.predict
        self.error
        self.optimize
        #self.model_learn

    @lazy_property
    def length(self):
        used = tf.sign(tf.reduce_max(tf.abs(self.data), axis=2))
        length = tf.reduce_sum(used, axis=1)
        length = tf.cast(length, tf.int32)
        return length

    @lazy_property
    def predict(self):
        # Recurrent network.
        length_ = self.length
        output, _ = rnn.dynamic_rnn(
            rnn_cell.GRUCell(self._num_hidden),
            self.data,
            dtype=tf.float32,
            sequence_length=length_,
        )
        # Softmax layer.
        max_length = int(self.target.get_shape()[1])
        num_classes = int(self.target.get_shape()[2])
        weight, bias = self._weight_and_bias(self._num_hidden, num_classes)
        # Flatten to apply same weights to all time steps.
        output = tf.reshape(output, [-1, self._num_hidden])
        prediction = tf.nn.softmax(tf.matmul(output, weight) + bias)
        prediction = tf.reshape(prediction, [-1, max_length, num_classes])
        return prediction
    
#    @lazy_property
#    def model_learn(self):
#        # Recurrent network.
#        length_ = self.length
#        output, _ = rnn.dynamic_rnn(
#            rnn_cell.GRUCell(self._num_hidden),
#            self.data,
#            dtype=tf.float32,
#            sequence_length=length_,
#        )
#        # Softmax layer.
#        max_length = int(self.target.get_shape()[1])
#        num_classes = int(self.target.get_shape()[2])
#        weight, bias = self._weight_and_bias(self._num_hidden, num_classes)
#        # Flatten to apply same weights to all time steps.
#        output = tf.reshape(output, [-1, self._num_hidden])
#        prediction = tf.nn.softmax(tf.matmul(output, weight) + bias)
#        prediction = tf.reshape(prediction, [-1, max_length, num_classes])
#        return prediction

    @lazy_property
    def cost(self):
        # Compute cross entropy for each frame.
        cross_entropy = self.target * tf.log(self.predict)
        cross_entropy = -tf.reduce_sum(cross_entropy, axis=2)
        mask = tf.sign(tf.reduce_max(tf.abs(self.target), axis=2))
        cross_entropy *= mask
        # Average over actual sequence lengths.
        cross_entropy = tf.reduce_sum(cross_entropy, axis=1)
        cross_entropy /= tf.cast(self.length, tf.float32)
        return tf.reduce_mean(cross_entropy)

    @lazy_property
    def optimize(self):
        learning_rate = 0.0003
        optimizer = tf.train.AdamOptimizer(learning_rate)
        return optimizer.minimize(self.cost)

    @lazy_property
    def error(self):
        y_act  = tf.argmax(self.target, 2)
        y_pred = tf.argmax(self.predict, 2)
        mistakes = tf.not_equal(y_act, y_pred)
        mistakes = tf.cast(mistakes, tf.float32)
        mask = tf.sign(tf.reduce_max(tf.abs(self.target), axis=2))
        mistakes *= mask
        # Average over actual sequence lengths.
        mistakes = tf.reduce_sum(mistakes, axis=1)
        mistakes /= tf.cast(self.length, tf.float32)
        return tf.reduce_mean(mistakes), y_act, y_pred, mask

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)


def get_dataset():
    """Read dataset and flatten images."""
    dataset = sets.Ocr()
    dataset = sets.OneHot(dataset.target, depth=2)(dataset, columns=['target'])
    dataset['data'] = dataset.data.reshape(
        dataset.data.shape[:-2] + (-1,)).astype(float)
    train, test = sets.Split(0.66)(dataset)
    return train, test


if __name__ == '__main__':
    train, test = get_dataset()
    y_act  = test.target
    _, length, image_size = train.data.shape
    num_classes = train.target.shape[2]
    data = tf.placeholder(tf.float32, [None, length, image_size])
    target = tf.placeholder(tf.float32, [None, length, num_classes])
    model = VariableSequenceLabelling(data, target)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    for epoch in range(10):
        for _ in range(100):
            batch = train.sample(10)
            sess.run(model.optimize, {data: batch.data, target: batch.target})
        error, y_act, y_pred, mask = sess.run(model.error, {data: test.data, target: test.target})
        print('Epoch {:2d} error {:3.1f}%'.format(epoch + 1, 100 * error))
        
        # y_pred, y_last = sess.run(model.prediction, {data: test.data, target: test.target})
        #y_pred = sess.run(model.predict, {data: test.data, target: test.target})
        #error  = (np.argmax(y_act, 2) != np.argmax(y_pred, 2)).astype(np.float32)
        #print(50*'=')
        #print('Epoch {:2d} error {:3.1f}%'.format(epoch + 1, 100 * error))
        #print(50*'=')

mask

