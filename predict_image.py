import get_data as gd
import tensorflow as tf
import numpy as np

X, Y = gd.get_photo_data(1, 1)
X = np.array(X)

def forward(X, W1, b1, W2, b2):
    """
    Operation to predict
    """
    Z = tf.nn.sigmoid(tf.matmul(X, W1) + b1)
    return tf.matmul(Z,W2)+b2

sess = tf.Session()
saver = tf.train.import_meta_graph('model.meta')
saver.restore(sess, tf.train.latest_checkpoint('./'))

W1 = sess.run('w1:0').astype(float).T
b1 = sess.run('b1:0').astype(float).T
W2 = sess.run('w2:0').astype(float).T
b2 = sess.run('b2:0').astype(float).T

y = tf.nn.softmax(forward(X[0], W1, b1, W2, b2))
