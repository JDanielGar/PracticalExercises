import tensorflow as tf
import numpy as np
from get_data import get_photo_data


X, Y = get_photo_data(2, 10)

X = np.array(X).astype(float)


N = len(Y)
D = 2304
M = 3523
K = 3

T = np.zeros((N, K))
for i in range(N):
    T[i, Y[i]] = 1

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def forward(X, W1, b1, W2, b2):
    Z = tf.nn.sigmoid(tf.matmul(X, W1) + b1)
    return tf.matmul(Z,W2)+b2

tfX = tf.placeholder(tf.float32, [None, D])
tfY = tf.placeholder(tf.float32, [None, K])

W1 = init_weights([D, M])
b1 = init_weights([M])
W2 = init_weights([M, K])
b2 = init_weights([K])

logits = forward(tfX, W1, b1, W2, b2)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tfY, logits=logits))

train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
predict_op = tf.arg_max(logits, 1)


# Train Part

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

for i in range(1000):
    sess.run(train_op, feed_dict={tfX: X, tfY: T})
    pred = sess.run(predict_op, feed_dict={tfX: X, tfY: T})
    if i % 100 == 0:
        print("Accuracy:", np.mean(Y == pred))

save_model = tf.train.Saver()
save_model.save(sess, 'model.ckpt')