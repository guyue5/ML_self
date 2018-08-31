import tensorflow as tf
import numpy as np


def linear_func(X,w,b):
    y_predict= tf.matmul(w, X) + b
    return y_predict

def loss_func(Y, y_predict):
    loss=tf.reduce_mean(tf.square(Y - y_predict),name='loss')
    # loss=tf.reduce_sum(tf.square(Y - y_predict),name='loss')
    # when use reduce_sum, learning_rate should be set <0.1, to keep the same convergence rate
    return loss

def main():

    train_epochs = 1000
    learning_rate=0.1

    train_x = np.float32(np.random.rand(2, 100))
    train_y = np.dot([0.100, 0.200], train_x) + 0.300

    X=tf.placeholder(tf.float32)
    Y=tf.placeholder(tf.float32)
    # w=tf.Variable(0.0,name="weights")
    # b=tf.Variable(0.0,name="biases")
    b = tf.Variable(tf.zeros([1]))
    w = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))

    y_predict=linear_func(X,w,b)
    loss=loss_func(Y,y_predict)
    train_op=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    init=tf.global_variables_initializer()
    # init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(train_epochs):
            _,train_loss=sess.run([train_op,loss],feed_dict={X:train_x,Y:train_y})
            if epoch % 10 == 0:
                print("epoch: %d, loss: %.5f, w: %s, b:  %s" %
                      (epoch,train_loss,sess.run(w),sess.run(b)))




if __name__ == '__main__':
   main()

