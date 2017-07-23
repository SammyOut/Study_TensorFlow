import tensorflow as tf

# x, y 데이터
x_data = [1, 2, 3]
y_data = [1, 2, 3]

# tf에서 사용할 변수 선언
W = tf.Variable(tf.random_normal([1]), name='weight')
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# 가설 : WX (y = ax)
hypothesis = X * W
# cost = (hypothesis - Y)^2 / m
cost = tf.reduce_sum(tf.square(hypothesis - Y))

# cost를 최소화 하기 위한 Gradient Descent algorithm
learning_rate = 0.1
gradient = tf.reduce_mean((W * X - Y) * X)
descent = W - learning_rate * gradient
update = W.assign(descent)

# Session을 열어 그래프를 실행
sess = tf.Session()
# tf가 사용할 변수들을 초기화
sess.run(tf.global_variables_initializer())
for step in range(21) :
    # x_data와 y_data를 제공하고 update 실행
    sess.run(update, feed_dict={X : x_data, Y : y_data})
    print(step, sess.run(cost, feed_dict={X : x_data, Y : y_data}), sess.run(W))